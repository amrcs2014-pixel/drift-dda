"""Wrap every \\input{tab_*.tex} in the manuscript sections with adjustbox (shrink only if wider than the text)."""
import glob, os, re

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DRIFT_DDA_v7', 'src')
pat = re.compile(r'^\\input\{(tab_[a-z_]+)\.tex\}\s*$', re.M)
for f in glob.glob(os.path.join(SRC, 'sec_*.tex')):
    s = open(f, encoding='utf-8').read()
    t = pat.sub(lambda m: '\\adjustbox{max width=\\linewidth}{\\input{%s.tex}}' % m.group(1), s)
    if t != s:
        open(f, 'w', encoding='utf-8').write(t)
        print('wrapped tables in', os.path.basename(f))
