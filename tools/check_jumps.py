import csv, math
from collections import defaultdict
fn='figs/boundaries_probs_merton_2x2.csv'
scenes=defaultdict(list)
with open(fn,'r') as f:
    r=csv.DictReader(f)
    for row in r:
        sc=row['scenario']
        t=float(row['t'])
        p=float(row['p_marg'])
        scenes[sc].append((t,p,row))
for sc,vals in scenes.items():
    print('---',sc)
    times=[item[0] for item in vals]
    ps=[max(1e-300,item[1]) for item in vals]
    ln=[math.log(p) for p in ps]
    diffs=[ln[i+1]-ln[i] for i in range(len(ln)-1)]
    jumps=[i for i,d in enumerate(diffs) if d>=10]
    print('selected t_star_prob=', vals[0][2]['t_star_prob'], 't_star_bs=', vals[0][2]['t_star_bs'], 't_star_drop=', vals[0][2]['t_star_drop'])
    print('jumps indices:', jumps)
    for j in jumps[:10]:
        print(' jump at', times[j], '->', times[j+1], ' diff=', diffs[j])
    if not jumps:
        print('no jumps >=10; last t=', times[-1])
