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


def evaluate(counts, risk):
    if counts.shape != risk.shape or counts.ndim != 1 or not len(risk):
        raise ValueError("Scores and labels must have matching non-empty rows")
    if not torch.isfinite(risk).all() or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Scores and labels must have matching non-empty finite rows")
    unsafe = counts > 0
    capture = float(counts[selected(risk,0.05)].sum())/float(counts.sum()) if counts.sum()>0 else float("nan")
    if unsafe.all() or not unsafe.any():
        return {"Capture@5%":capture,"AUROC":float("nan"),"AUPRC":float("nan"),"NAURC":float("nan")}
    return {"Capture@5%":capture,"AUROC":binary_auroc(unsafe,risk),"AUPRC":average_precision(unsafe,risk),"NAURC":normalized_aurc(unsafe,risk)[1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("fn", "fp", "both"), default="both")
    parser.add_argument("--statistics",type=Path,default=Path("statistics"))
    parser.add_argument("--models",type=Path)
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/concepts.json")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,default=Path("outputs/risk_metrics.csv"))
    args=parser.parse_args();torch.set_num_threads(4)
    rows=[]
    paths=sorted((args.statistics/"targets").glob("*.pt"))
    if not paths: parser.error("No target statistics found")
    for path in paths:
        payload=load_tensor_file(path)
        for task in (("fn", "fp") if args.task=="both" else (args.task,)):
            count_key="false_positive_count" if task=="fp" else "miss_count"
            score_key="fp_scores" if task=="fp" else "scores"
            methods={"ours_detsae":payload[score_key]["ours_detsae"]}
            if args.models:
                config=json.loads(args.config.read_text())
                raw=predict_risk(payload["concept"],args.models,config,args.device,task=task)
                order=torch.argsort(raw,stable=True); rank=torch.empty_like(raw)
                rank[order]=torch.arange(len(raw))/max(1,len(raw)-1)
                methods["ours_recomputed"]=rank
            for name, score in methods.items():
                row={"dataset":path.stem,"task":task,"method":name,**evaluate(payload[count_key],score)}
                rows.append(row)
                print(path.stem,task,name," ".join(f"{k}={100*v:.2f}" for k,v in row.items() if isinstance(v,float)),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    main()
