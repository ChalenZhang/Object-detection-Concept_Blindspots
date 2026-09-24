"""Evaluate image risk using locally supplied detection annotations."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concept_blindspots.model import load_tensor_file
from concept_blindspots.matching import error_counts
from evaluate_statistics import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--task",choices=("fn","fp"))
    parser.add_argument("--kitti-labels",type=Path)
    parser.add_argument("--groups",choices=["car-person","car"],default="car-person")
    parser.add_argument("--output",type=Path,default=Path("outputs/evaluation.json"))
    args=parser.parse_args()
    predictions=load_tensor_file(args.predictions)
    task=args.task or predictions.get("task","fn")
    if predictions.get("task","fn")!=task:
        parser.error("Prediction task and evaluation task differ")
    if not (len(predictions["images"])==len(predictions["detections"])==len(predictions["risk"])):
        raise ValueError("Prediction images, detections, and risks must have equal lengths")
    names=[Path(name).name for name in predictions["images"]]
    if len(set(names))!=len(names):raise ValueError("Prediction image names must be unique")
    labels=json.loads(args.annotations.read_text())
    image_names={int(r["id"]):Path(r["file_name"]).name for r in labels["images"]}
    if len(set(image_names.values()))!=len(image_names):raise ValueError("Annotation image names are not unique")
    class_names=["background","bicycle","bus","car","motorcycle","person","rider","truck"]
    aliases={"pedestrian":"person","psn.":"person","psn":"person","person_sitting":"person","cyclist":"rider","bike":"bicycle","motor":"motorcycle"}
    categories={}
    for row in labels["categories"]:
        name=row["name"].strip().lower()
        if task=="fp" and args.kitti_labels and name=="person_sitting":continue
        name=aliases.get(name,name)
        if name in class_names:categories[int(row["id"])]=class_names.index(name)
    required=(3,) if args.groups=="car" else (3,5)
    if not set(required)<=set(categories.values()):raise ValueError("Required evaluation categories are absent from annotations")
    by_image=defaultdict(list)
    for row in labels["annotations"]:
        class_id=categories.get(int(row["category_id"]))
        if class_id is None or row.get("iscrowd",0):continue
        x,y,w,h=map(float,row["bbox"])
        by_image[image_names[int(row["image_id"])]].append({"class_id":class_id,"box":[x,y,x+w,y+h]})
    counts=[]
    allowed=set(image_names.values())
    for name,prediction in zip(predictions["images"],predictions["detections"]):
        if Path(name).name not in allowed:raise ValueError(f"Missing annotation image: {name}")
        original=None
        if args.kitti_labels and task=="fp":
            original=[]
            path=args.kitti_labels/(Path(name).stem+".txt")
            for line in path.read_text().splitlines():
                if not line.strip():continue
                fields=line.split()
                if len(fields)<8:raise ValueError(f"Invalid KITTI annotation in {path.name}")
                original.append({"type":fields[0],"bbox_xyxy":[float(v) for v in fields[4:8]]})
        counts.append(error_counts(by_image[Path(name).name],prediction,required,task,original))
    risk=predictions["risk"].float()
    ranks=torch.empty_like(risk);ranks[torch.argsort(risk,stable=True)]=torch.arange(len(risk))/max(1,len(risk)-1)
    result={"task":task,"images":len(counts),"error_count":int(sum(counts)),**evaluate(torch.tensor(counts),ranks)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
