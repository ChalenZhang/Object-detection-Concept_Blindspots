"""Refit all concept-cluster effects and the joint blindspot test."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concept_blindspots.model import load_tensor_file
from concept_blindspots.discovery_utils import fit_ridge, nanmean, leave_one_style_out, clustered_bootstrap_coefficients, centered_bootstrap_pvalue, adjust_fdr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("statistics/discovery.pt"))
    parser.add_argument("--output",type=Path,default=Path("outputs/blindspots.csv"))
    parser.add_argument("--bootstrap",type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<100:parser.error("Use at least 100 bootstrap replicates")
    torch.set_num_threads(4)
    results=[]
    for class_id,p in load_tensor_file(args.input).items():
        design=p["design"].numpy();valid=p["valid"].bool().numpy()
        loss=(p["origin_quality"][:,None]-p["styled_quality"]).numpy()
        centered=loss-nanmean(loss,valid,axis=0)[None,:]
        outcome=nanmean(centered,valid,axis=1)
        keep=np.isfinite(outcome);design=design[keep];outcome=outcome[keep]
        start=p["cluster_start"];fold=p["cv_fold"].numpy()[keep]
        candidates=[]
        for alpha in [0,1e-5,1e-4,1e-3,1e-2,1e-1]:
            errors=[]
            for f in range(5):
                test=fold==f
                if not test.any() or (~test).sum()<=design.shape[1]:continue
                beta=fit_ridge(design[~test],outcome[~test],start,alpha)
                errors.extend(((outcome[test]-design[test]@beta)**2).tolist())
            if errors:candidates.append((float(np.mean(errors)),alpha))
        alpha=min(candidates)[1]
        beta=fit_ridge(design,outcome,start,alpha)[start:]
        scenes=[f"{int(i):08d}" for i in p["scene"][keep]]
        bootstrap=clustered_bootstrap_coefficients(design,outcome,scenes,start,alpha,args.bootstrap,200,2027+int(class_id)*1000)
        lower,upper=np.quantile(bootstrap,[0.025,0.975],axis=0)
        omitted=leave_one_style_out(design,centered[keep],valid[keep],start,alpha).min(axis=0)
        for i,name in enumerate(p["cluster_ids"]):
            results.append({"cluster_id":name,"class_id":int(class_id),"excess_q50_loss":float(beta[i]),"excess_q50_loss_ci_low":float(lower[i]),"excess_q50_loss_ci_high":float(upper[i]),"loss_above_practical_threshold_pvalue":centered_bootstrap_pvalue(bootstrap[:,i],beta[i],0.05),"leave_one_style_out_min_effect":float(omitted[i])})
        print(f"class={class_id} clusters={len(beta)} ridge={alpha:g}",flush=True)
    adjust_fdr(results,"loss_above_practical_threshold_pvalue","loss_above_practical_threshold_global_fdr")
    for row in results:
        row["is_task_transfer_blindspot"]=int(row["excess_q50_loss"]>0.05 and row["excess_q50_loss_ci_low"]>0.05 and row["loss_above_practical_threshold_global_fdr"]<=0.05 and row["leave_one_style_out_min_effect"]>0.05)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(f"blindspots={sum(row['is_task_transfer_blindspot'] for row in results)}/{len(results)}")


if __name__=="__main__":
    main()
