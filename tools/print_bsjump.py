import csv
import sys
fn = sys.argv[1] if len(sys.argv) > 1 else 'figs/boundaries_probs_merton_2x2_bsjump.csv'
seen = {}
with open(fn, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        sc = row['scenario']
        t = row.get('t_star_bs_jump')
        if t and t.strip() != '':
            seen.setdefault(sc, (t, row.get('b_star_bs_jump'), row.get('p_star_bs_jump')))
# print results
for sc in ['A_call','B_call','A_put','B_put']:
    v = seen.get(sc)
    if v:
        print(f"{sc}: t_bs_jump={v[0]}, b_bs_jump={v[1]}, p_bs_jump={v[2]}")
    else:
        print(f"{sc}: no BS ln-jump (red X) found in CSV")
