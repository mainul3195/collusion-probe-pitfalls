import json,glob,sys,numpy as np
from sklearn.metrics import roc_auc_score
rows=[]
for f in sorted(glob.glob(sys.argv[1]+"/*.json")):
    r=json.load(open(f)); c=json.load(open(glob.glob("runs/qwen3_14b/*/%s/run_config.json"%r["run_id"])[0]))
    S=np.array([r["scores"][a] for a in r["agents"]]); col=[a in c["coalition"] for a in r["agents"]] if c["mode"]=="collusion" else [False]*len(r["agents"])
    rows.append((c["mode"],c.get("collusion_success"),c["k"],c["onset"],r["any"],S,np.array(col)))
print(len(rows))
for m,su,k,o,anyc,S,col in rows[:int(sys.argv[2])]:
    print(m[:4],"succ" if su else "    ","k",k,"on",o,"any",anyc,"| colluders last",S[col,-1].tolist() if col.any() else "-","| others last",S[~col,-1].tolist())
sel=[r for r in rows if r[0]!="collusion" or r[1]]
y=[1 if r[0]=="collusion" else 0 for r in sel]
for name,f in [("peak-last",lambda S:S[:,-1].max()),("peak-mean",lambda S:S.mean(1).max()),("spread-last",lambda S:S[:,-1].max()-np.median(S[:,-1]))]:
    s=[f(r[5]) for r in sel]
    print("pos",sum(y),"neg",len(y)-sum(y),name,"AUROC succ vs rest", round(roc_auc_score(y,s),3) if 0<sum(y)<len(y) else None)
print("mean any (last) control:", np.mean([r[4][-1] for r in rows if r[0]=="control"]), "succ:", np.mean([r[4][-1] for r in rows if r[0]=="collusion" and r[1]]))
