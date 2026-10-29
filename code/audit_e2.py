import json,glob,collections,sys
for raw in sys.argv[1:]:
    rows=[json.load(open(f)) for f in glob.glob(raw+'/*/run_config.json')]
    col=[r for r in rows if r['mode']=='collusion']
    print(raw, 'runs',len(rows), collections.Counter(r['mode'] for r in rows))
    s=sum(r['collusion_success'] for r in col); m=sum(r['target_majority'] for r in col)
    print(f' collusion success {s}/{len(col)}  target_majority {m}/{len(col)}')
    for key in ['k','n','onset']:
        d=collections.defaultdict(lambda:[0,0])
        for r in col: d[r[key]][0]+=r['collusion_success']; d[r[key]][1]+=1
        print(' by',key, {a:f'{b[0]}/{b[1]}' for a,b in sorted(d.items())})
    for md in ['benign_coord','control']:
        x=[r for r in rows if r['mode']==md]
        print(f' {md}: target_majority {sum(r["target_majority"] for r in x)}/{len(x)}')
