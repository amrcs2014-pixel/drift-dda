"""Self-healing experiment queue (launched every 10 min by a scheduled task; exits at once if another
instance holds the lock). Order: tuning -> ToN cache -> experiments. Every stage is resumable."""
import os, sys, time, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
LOCK = os.path.join(RES, 'queue.lock')
LOG = os.path.join(RES, 'queue.log')
PY = sys.executable.replace('pythonw.exe', 'python.exe')   # the task runs pythonw (no window); stages use python
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)           # hidden console, inherited by worker processes
WORKERS = int(os.environ.get('DDA_WORKERS', '6'))
STAGES = ['real', 'heldout', 'rotation', 'synth', 'ablation', 'injected', 'poison', 'severity', 'frontier', 'sensitivity', 'posthoc']

def log(msg):
    with open(LOG, 'a') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))

def pid_alive(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH'], capture_output=True, text=True, creationflags=NOWIN).stdout
        return str(pid) in out
    except Exception:
        return False

def acquire():
    os.makedirs(RES, exist_ok=True)
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip() or 0)
        except Exception:
            pid = 0
        if pid and pid_alive(pid):
            return False
    open(LOCK, 'w').write(str(os.getpid()))
    return True

def run(args, logname):
    with open(os.path.join(RES, logname), 'a') as f:
        f.write('\n==== %s %s\n' % (time.ctime(), ' '.join(args))); f.flush()
        return subprocess.call([PY, '-u'] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, creationflags=NOWIN)

def other_runner_alive():
    """True if a tune/experiments runner started outside this queue is still running."""
    try:
        cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }")
        out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, creationflags=NOWIN).stdout
    except Exception:
        return False
    for line in out.splitlines():
        pid, _, cl = line.partition('|')
        cl = cl.strip()
        if pid.strip() == str(os.getpid()):
            continue
        if cl.endswith(('tune.py run 5', 'tune.py run 4')) or ' experiments.py ' in (cl + ' '):
            return True
    return False

def main():
    done = [os.path.join(RES, f) for f in ['QUEUE_DONE', 'cost_cpu1.json', 'cost_cpu6.json', 'cost_gpu.json']]
    if all(os.path.exists(f) for f in done):
        return
    if other_runner_alive():
        return
    if not acquire():
        return
    log('queue start pid %d' % os.getpid())
    try:
        if not os.path.exists(os.path.join(RES, 'tuning_selected.json')):
            log('tuning'); run(['tune.py', 'run', str(WORKERS)], 'tune_run.log')
        if not os.path.exists(os.path.join(HERE, '..', 'data', 'ton', 'meta.json')):
            log('prep ton'); run(['prep_real.py', 'ton'], 'prep_ton.log')
        failed = []
        for st in STAGES:
            log('stage %s' % st)
            rc = run(['experiments.py', st, str(WORKERS)], 'exp_%s.log' % st)
            log('stage %s rc=%s' % (st, rc))
            if rc != 0:
                failed.append(st)
        if failed:                     # killed (reboot/logoff) or crashed: the next scheduled run resumes
            log('stages not completed: %s' % failed)
            return
        for dev, thr, out in [('cpu', 1, 'cost_cpu1.json'), ('cpu', 6, 'cost_cpu6.json'), ('cuda', 1, 'cost_gpu.json')]:
            if not os.path.exists(os.path.join(RES, out)):
                log('cost %s %d' % (dev, thr))
                run(['cost_bench.py', '--device', dev, '--threads', str(thr), '--out', os.path.join(RES, out)], 'cost.log')
        run(['aggregate.py'], 'aggregate.log')
        log('all stages done')
        open(os.path.join(RES, 'QUEUE_DONE'), 'w').write(time.ctime())
    finally:
        try:
            os.remove(LOCK)
        except Exception:
            pass

if __name__ == '__main__':
    main()
