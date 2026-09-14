"""Recompute risk-ranking results from released per-image statistics."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concept_blindspots.metrics import binary_auroc, average_precision, normalized_aurc, selected
from concept_blindspots.model import load_tensor_file, predict_risk


def evaluate(misses, risk):
    if len(misses) != len(risk) or not torch.isfinite(risk).all() or not len(risk):
        raise ValueError("Scores and labels must have matching non-empty finite rows")
    unsafe = misses > 0
    capture = float(misses[selected(risk,0.05)].sum()/misses.sum()) if misses.sum()>0 else float("nan")
    return {"Capture@5%":capture,"AUROC":binary_auroc(unsafe,risk),"AUPRC":average_precision(unsafe,risk),"NAURC":normalized_aurc(unsafe,risk)[1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics",type=Path,default=Path("statistics"))
    parser.add_argument("--models",type=Path)
    parser.add_argument("--config",type=Path,default=Path("configs/concepts.json"))
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,default=Path("outputs/risk_metrics.csv"))
    args=parser.parse_args();torch.set_num_threads(4)
    rows=[]
    paths=sorted((args.statistics/"targets").glob("*.pt"))
    if not paths: parser.error("No target statistics found")
    for path in paths:
        payload=load_tensor_file(path)
        methods=dict(payload["scores"])
        if args.models:
            config=json.loads(args.config.read_text())
            raw=predict_risk(payload["concept"],args.models,config,args.device)
            # Match the empirical percentile score used in the benchmark.
            order=torch.argsort(raw,stable=True); rank=torch.empty_like(raw)
            rank[order]=torch.arange(len(raw))/max(1,len(raw)-1)
            methods["ours_recomputed"]=rank
        for name, score in methods.items():
            row={"dataset":path.stem,"method":name,**evaluate(payload["miss_count"],score)}
            rows.append(row)
            print(path.stem,name," ".join(f"{k}={100*v:.2f}" for k,v in row.items() if isinstance(v,float)),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
