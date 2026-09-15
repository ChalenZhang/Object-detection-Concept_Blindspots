"""Summarize every recorded cluster and matched intervention control."""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(scene_values, samples):
    values=np.asarray([scene_values[s] for s in sorted(scene_values)],dtype=np.float64)
    counts=values[:,0];sums=values[:,1:]
    mean=sums.sum(axis=0)/counts.sum()
    if len(counts)<2 or not samples:return mean,np.full(2,np.nan),np.full(2,np.nan)
    rng=np.random.default_rng(2027);parts=[]
    for start in range(0,samples,128):
        weights=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=min(128,samples-start))
        parts.append((weights@sums)/(weights@counts)[:,None])
    lower,upper=np.quantile(np.concatenate(parts),[0.025,0.975],axis=0)
    return mean,lower,upper


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics",type=Path,default=Path("statistics"))
    parser.add_argument("--output",type=Path,default=Path("outputs/interventions.csv"))
    parser.add_argument("--bootstrap",type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap<1:parser.error("bootstrap must be positive")
    paths=sorted((args.statistics/"interventions").glob("*.csv.gz"))
    if not paths:parser.error("No intervention records found")
    rows=[]
    for path in paths:
        aggregates=defaultdict(lambda:defaultdict(lambda:np.zeros(3)))
        clusters=defaultdict(set)
        with gzip.open(path,"rt") as handle:
            for record in csv.DictReader(handle):
                if record["mechanism"] not in {"excess","missing"}:continue
                delta=float(record["target_margin_after"])-float(record["target_margin_before"])
                values=np.array([1,delta,float(record["top1_gain"])])
                base=(record["class_id"],record["mechanism"],record["intervention"],float(record["dose"]))
                for name in ["all",record["cluster_id"]]:
                    key=base+(name,)
                    aggregates[key][int(record["scene"])]+=values
                    clusters[key].add(record["cluster_id"])
        for key,scenes in aggregates.items():
            cls,mechanism,intervention,dose,cluster=key
            repeats=args.bootstrap if dose==1.0 and intervention=="paired_source_restore" else 0
            mean,low,high=summarize(scenes,repeats)
            rows.append({"detector":path.name.split('.')[0].upper(),"class_id":int(cls),"mechanism":mechanism,"intervention":intervention,"dose":dose,"cluster_id":cluster,"clusters":len(clusters[key]),"events":int(sum(v[0] for v in scenes.values())),"margin_gain":mean[0],"margin_ci_low":low[0],"margin_ci_high":high[0],"top1_net_recovery":mean[1],"top1_ci_low":low[1],"top1_ci_high":high[1]})
        print(f"{path.name}: {len(aggregates)} groups",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":main()
