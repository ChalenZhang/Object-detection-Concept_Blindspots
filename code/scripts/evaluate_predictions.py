"""Evaluate image risk using locally supplied detection annotations."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concept_blindspots.model import load_tensor_file
from concept_blindspots.matching import match_group
from evaluate_statistics import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--groups",choices=["car-person","car"],default="car-person")
    parser.add_argument("--output",type=Path,default=Path("outputs/evaluation.json"))
    args=parser.parse_args()
    predictions=load_tensor_file(args.predictions)
    labels=json.loads(args.annotations.read_text())
    image_names={int(r["id"]):Path(r["file_name"]).name for r in labels["images"]}
    if len(set(image_names.values()))!=len(image_names):raise ValueError("Annotation image names are not unique")
    class_names=["background","bicycle","bus","car","motorcycle","person","rider","truck"]
    aliases={"pedestrian":"person","person_sitting":"person","cyclist":"rider","bike":"bicycle","motor":"motorcycle"}
    categories={}
    for row in labels["categories"]:
        name=row["name"].lower()
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
    misses=[]
    allowed=set(image_names.values())
    for name,prediction in zip(predictions["images"],predictions["detections"]):
        if Path(name).name not in allowed:raise ValueError(f"Missing annotation image: {name}")
        result=match_group(by_image[Path(name).name],prediction,required,0.05,0.5,100)
        misses.append(result["miss_count"])
    risk=predictions["risk"].float()
    ranks=torch.empty_like(risk);ranks[torch.argsort(risk,stable=True)]=torch.arange(len(risk))/max(1,len(risk)-1)
    result=evaluate(torch.tensor(misses),ranks)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
