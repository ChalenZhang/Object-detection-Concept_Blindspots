"""Train the risk MLP from source image-level statistics."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concept_blindspots.model import RiskMLP, risk_inputs, load_tensor_file
from concept_blindspots.losses import streaming_moments, ranking_loss
from concept_blindspots.training_utils import SceneViewStream, cosine_learning_rate, calibration_metrics, selection_key

GROUPS = ["car", "person", "vehicle", "people", "all"]


@torch.inference_mode()
def score(model, inputs, device):
    model.eval()
    parts=[]
    for begin in range(0,len(inputs),256):
        output=model(inputs[begin:begin+256].to(device))
        count=torch.expm1(output["log_missed_count"].clamp(-8,8)).clamp_min(0)
        parts.append(count[:,:2].sum(dim=1).cpu())
    return torch.cat(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics",type=Path,default=Path("statistics"))
    parser.add_argument("--configs",type=Path,default=Path("configs"))
    parser.add_argument("--output",type=Path,default=Path("outputs/retrained_models"))
    parser.add_argument("--seeds",type=int,nargs="+",default=[2027,2028,2029])
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--steps",type=int)
    args=parser.parse_args();torch.set_num_threads(4)
    train=load_tensor_file(args.statistics/"source_train.pt")
    calibration=load_tensor_file(args.statistics/"source_calibration.pt")
    definition=json.loads((args.configs/"concepts.json").read_text())
    conf=json.loads((args.configs/"training.json").read_text())
    steps=args.steps or conf["max_steps"]
    if steps<1:parser.error("steps must be positive")
    checkpoints=sorted(set([s for s in conf["checkpoint_steps"] if s<=steps]+[steps]))
    inputs=risk_inputs(train["concept"],definition)
    cal_inputs=risk_inputs(calibration["concept"],definition)
    targets=train["targets"]
    cal_miss=calibration["targets"]["miss_count"][:,:2].sum(dim=1)
    cal_labels={"miss_count":cal_miss,"unsafe":cal_miss>0}
    mean,std=streaming_moments(inputs,torch.arange(len(inputs)))
    scene_rows=[torch.nonzero(train["scene"]==s).flatten() for s in sorted(train["scene"].unique().tolist())]
    positive=targets["has_miss"].sum(dim=0)
    pos_weight=((len(inputs)-positive)/positive.clamp_min(1)).clamp(0.25,20).to(args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        candidates=[]
        for multiplier in conf["lr_multipliers"]:
            random.seed(seed);torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            model=RiskMLP(mean,std,conf["hidden_dim"],conf["dropout"]).to(args.device)
            learning_rate=conf["base_learning_rate"]*multiplier
            optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=conf["weight_decay"])
            stream=SceneViewStream(scene_rows,conf["effective_batch_size"],seed+8191)
            best=None;history=[]
            for step in range(1,steps+1):
                lr=cosine_learning_rate(learning_rate,step,steps)
                for group in optimizer.param_groups:group["lr"]=lr
                index=stream.draw()
                x=inputs[index].to(args.device)
                binary=targets["has_miss"][index].to(args.device)
                count=torch.log1p(targets["miss_count"][index].to(args.device))
                quality=torch.log1p(targets["q50_loss"][index].to(args.device))
                model.train();output=model(x)
                loss=F.binary_cross_entropy_with_logits(output["binary_logits"],binary,pos_weight=pos_weight)
                loss=loss+F.smooth_l1_loss(output["log_missed_count"],count)
                loss=loss+conf["q50_weight"]*F.smooth_l1_loss(output["log_q50"],quality)
                rank=torch.stack([ranking_loss(output["log_missed_count"][:,i],count[:,i]) for i in range(len(GROUPS))]).mean()
                loss=loss+conf["ranking_weight"]*rank
                if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
                if step not in checkpoints:continue
                metrics=calibration_metrics(cal_labels,score(model,cal_inputs,args.device))
                history.append({"step":step,"loss":float(loss.detach()),**metrics})
                key=selection_key(metrics,step)
                if best is None or key>best[0]:
                    best=(key,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},step)
                print(f"seed={seed} lr_multiplier={multiplier} step={step}/{steps} capture5={metrics['capture05']:.6f}",flush=True)
            payload={"seed":seed,"groups":GROUPS,"dimensions":{"concept":inputs.shape[1]},"normalization":{"means":{"concept":mean},"stds":{"concept":std}},"model_config":{"hidden_dim":conf["hidden_dim"],"dropout":conf["dropout"]},"state_dict":best[1],"selected_step":best[2],"training_config":{"max_steps":steps,"effective_batch_size":conf["effective_batch_size"],"learning_rate":learning_rate,"weight_decay":conf["weight_decay"],"ranking_weight":conf["ranking_weight"],"q50_weight":conf["q50_weight"],"seed":seed}}
            candidates.append((best[0],-abs(math.log(multiplier)),payload))
            (args.output/f"training_seed_{seed}_lr_{multiplier:g}.json").write_text(json.dumps(history,indent=2)+"\n")
        chosen=max(candidates,key=lambda row:(row[0],row[1]))[2]
        torch.save(chosen,args.output/f"risk_seed_{seed}.pt")


if __name__=="__main__":
    main()
