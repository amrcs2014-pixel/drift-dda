"""Build DRIFT_DDA_submission_v7: flattened manuscript, bib/bbl/PDF, figures, figure sources, code, results."""
import os, re, shutil, subprocess, zipfile, glob, json

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, '..')
SRC = os.path.join(WORK, 'DRIFT_DDA_v7', 'src')
OUT = os.path.join(WORK, 'DRIFT_DDA_submission_v7')

def flatten(path, seen=None):
    seen = seen or set()
    txt = open(path, encoding='utf-8').read()
    def rep(m):
        name = m.group(1)
        f = os.path.join(SRC, name if name.endswith('.tex') else name + '.tex')
        if not os.path.exists(f) or f in seen:
            return m.group(0)
        seen.add(f)
        body = flatten(f, seen).rstrip('\n')
        if os.path.basename(f).startswith('tab_') or os.path.basename(f) == 'numbers.tex':
            return body                              # inline (tables sit inside \adjustbox{...}{...})
        return '%% ---- begin ' + os.path.basename(f) + '\n' + body + '\n%% ---- end ' + os.path.basename(f) + '\n'
    return re.sub(r'\\input\{([^}]+)\}', rep, txt)

def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for d in ['src', 'figs', 'figs_src', 'code', 'code/results']:
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    flat = flatten(os.path.join(SRC, 'DRIFT_DDA_v7.tex'))
    flat = flat.replace('\\graphicspath{{./}{../figs/}{figs/}}', '\\graphicspath{{./}{../figs/}{figs/}}')
    open(os.path.join(OUT, 'src', 'DRIFT_DDA_v7.tex'), 'w', encoding='utf-8').write(flat)
    for f in ['references.bib']:
        shutil.copy(os.path.join(SRC, f), os.path.join(OUT, 'src', f))
    figs = sorted(glob.glob(os.path.join(WORK, 'DRIFT_DDA_v7', 'figs', '*.pdf')))
    used = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', flat))
    for f in figs:
        if os.path.basename(f) in used:
            shutil.copy(f, os.path.join(OUT, 'figs'))
    shutil.copy(os.path.join(WORK, 'DRIFT_DDA_v7', 'figs_src', 'fig1_architecture.tex'), os.path.join(OUT, 'figs_src'))
    # compile the flattened manuscript next to its figures to check it is self-contained
    build = os.path.join(OUT, '_build'); os.makedirs(build, exist_ok=True)
    shutil.copy(os.path.join(OUT, 'src', 'DRIFT_DDA_v7.tex'), build); shutil.copy(os.path.join(OUT, 'src', 'references.bib'), build)
    for f in glob.glob(os.path.join(OUT, 'figs', '*.pdf')):
        shutil.copy(f, build)
    for cmd in (['pdflatex', '-interaction=nonstopmode', 'DRIFT_DDA_v7.tex'], ['bibtex', 'DRIFT_DDA_v7'],
                ['pdflatex', '-interaction=nonstopmode', 'DRIFT_DDA_v7.tex'], ['pdflatex', '-interaction=nonstopmode', 'DRIFT_DDA_v7.tex']):
        subprocess.run(cmd, cwd=build, capture_output=True)
    log = open(os.path.join(build, 'DRIFT_DDA_v7.log'), encoding='latin-1').read()
    errs = [l for l in log.splitlines() if l.startswith('!')]
    undef = len(re.findall(r'undefined', log))
    shutil.copy(os.path.join(build, 'DRIFT_DDA_v7.pdf'), os.path.join(OUT, 'src'))
    shutil.copy(os.path.join(build, 'DRIFT_DDA_v7.bbl'), os.path.join(OUT, 'src'))
    shutil.copy(os.path.join(build, 'DRIFT_DDA_v7.pdf'), os.path.join(WORK, '..', 'DRIFT_DDA_v7.pdf'))
    shutil.rmtree(build)
    # code + results
    for f in glob.glob(os.path.join(HERE, '*.py')) + [os.path.join(HERE, 'README.md')]:
        if os.path.basename(f).startswith(('explore_', 'pilot_', 'diag_')) or os.path.basename(f) == 'package.py':
            dst = os.path.join(OUT, 'code', 'design_phase'); os.makedirs(dst, exist_ok=True); shutil.copy(f, dst)
        else:
            shutil.copy(f, os.path.join(OUT, 'code'))
    for f in ['results_v7.json', 'tuning_selected.json', 'data_table.json', 'diag_ton.json',
              'heldout_first_run_with_identifier.json', 'cost_cpu1.json', 'cost_cpu6.json', 'cost_gpu.json']:
        p = os.path.join(WORK, 'results', f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, 'code', 'results'))
    for f in ['CHANGES_v7.md', 'highlights.txt']:
        p = os.path.join(WORK, 'DRIFT_DDA_v7', f)
        if os.path.exists(p):
            shutil.copy(p, OUT)
    # zips: submission package and (separately) the per-run logs
    z = os.path.join(WORK, '..', 'DRIFT_DDA_submission_v7.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.join('DRIFT_DDA_submission_v7', os.path.relpath(full, OUT)))
    zr = os.path.join(WORK, '..', 'DRIFT_DDA_v7_run_logs.zip')
    with zipfile.ZipFile(zr, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(os.path.join(WORK, 'results', 'raw')):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.join('raw', os.path.relpath(full, os.path.join(WORK, 'results', 'raw'))))
    print('LaTeX errors:', errs[:5], '| "undefined" mentions:', undef)
    print('package:', z, os.path.getsize(z) // 1024, 'KB; run logs:', zr, os.path.getsize(zr) // (1024 * 1024), 'MB')

if __name__ == '__main__':
    main()
