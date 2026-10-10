import re
from collections import Counter

import sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
t = t[:t.rfind('References')]
# prose lines only: drop table/figure debris (short lines, mostly numbers)
lines = [l.strip() for l in t.splitlines()]
prose = ' '.join(l for l in lines if len(l.split()) >= 6 and sum(c.isalpha() for c in l) > 0.6 * len(l))
prose = re.sub(r'-\s+', '-', prose)
sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', prose) if len(s.split()) >= 4]
L = [len(s.split()) for s in sents]
import statistics as st
print(f'sentences {len(L)}; mean {st.mean(L):.1f}; median {st.median(L)}; SD {st.pstdev(L):.1f}; CV {st.pstdev(L)/st.mean(L):.2f}; share <= 12 words {sum(x <= 12 for x in L)/len(L):.2f}; share >= 30 {sum(x >= 30 for x in L)/len(L):.2f}')
op = Counter(' '.join(s.split()[:1]) for s in sents)
print('top sentence openers:', op.most_common(12))
low = prose.lower()
for w in ['we find', 'we ', 'our ', 'suggest', 'may ', 'needed', 'however', 'in contrast', 'on the other hand', 'significant', 'indeed', 'strongly', 'interpretation:', 'recommendation:', ':', ';', '(', '—', ' – ', 'e.g.', 'i.e.']:
    print(f'  {w!r}: {low.count(w)}')
us = sum(low.count(w) for w in ['analyz', 'characteriz', 'maximiz', 'minimiz', 'behavior', 'color', 'utiliz'])
uk = sum(low.count(w) for w in ['analys', 'characteris', 'maximis', 'minimis', 'behaviour', 'colour', 'labelled'])
print(f'American -ize/-yze/-or forms {us}; British forms {uk}')
runins = re.findall(r'(?:^|\n)([A-Z][A-Za-z&./\- ]{2,40}?)\. [A-Z]', t)
print('run-in labels:', runins[:20])
print('passive (rough):', len(re.findall(r'\b(is|are|was|were|be|been)\s+\w+ed\b', low)), 'per', len(L), 'sentences')
