import json
import pandas as pd
from datasets import DatasetDict, Dataset
import re

data = []
with open('train.json') as f:
    #data = f.readlines()
    data = json.load(f)

with open('rel_info_train.json') as f:
    rels = json.load(f)

print(len(data))
keys = list(data.keys())
print(keys)

inst = []
gt = []
for prop in keys:
    for i in range(len(data[prop])):
        instruction = ' '.join(data[prop][i]['tokens'])
        pattern = r'(\s*")\s*([^"]*?)\s*("\s*)'
        instruction = re.sub(pattern, r'\1\2\3', instruction)
        instruction = instruction.strip().replace(' ,',',').replace(' )', ')').replace('( ','(').replace(' .','.').replace(' - ','-').replace(" '","'")
        inst.append(instruction.strip())
        gt.append(f"({data[prop][i]['h'][0]} | {rels[prop]} | {data[prop][i]['t'][0]})")

data_pd = pd.DataFrame(zip(inst,gt),columns=['input', 'output'])
print(data_pd.head())
print(data_pd['input'][1000])
print(data_pd['output'][1000])
dataset = DatasetDict({"train": Dataset.from_pandas(data_pd)})
dataset.push_to_hub("UofA-LINGO/FewRel-train-noins", private=True)
#print(inst)
#print(gt)