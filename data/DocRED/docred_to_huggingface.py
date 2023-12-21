import json
import pandas as pd
from datasets import DatasetDict, Dataset
import re

data = []
with open('train_annotated.json', encoding='utf-8') as f:
    #data = f.readlines()
    data = json.load(f)

with open('rel_info_train.json', encoding='utf-8') as f:
    rels = json.load(f)

print(len(data))
print(data[0].keys())
print(data[0]['vertexSet'])
keys = ['vertexSet', 'labels', 'title', 'sents']
inst = []
gt = []
for i in range(len(data)):
    triples = set()
    instruction = ""
    cnt = 0
    for sent in data[i]['sents']:
        tmp = ' '.join(sent)
        pattern = r'(\s*")\s*([^"]*?)\s*("\s*)'
        tmp = re.sub(pattern, r'\1\2\3', tmp)
        instruction += " " + tmp.strip().replace(' ,',',').replace(' )', ')').replace('( ','(').replace(' .','.').replace(' - ','-').replace(" '","'")
    inst.append(instruction.strip())
    triples = set()
    for rel in data[i]['labels']:
        ev = rel['evidence']
        subs = []
        objs = []
        for sub in data[i]['vertexSet'][rel['h']]:
            #if sub['sent_id'] in ev:
            #if cnt in ev:
            subs.append(sub['name'])
        for obj in data[i]['vertexSet'][rel['t']]:
            #if obj['sent_id'] in ev:
            #if cnt in ev:
            objs.append(obj['name'])
        # Match all subjects and objects together
        for sub in subs:
            for obj in objs:
                #triples.add(f"({sub.replace(' ', '_')} | {rels[rel['r']]} | {obj.replace(' ', '_')})")
                triples.add(f"({sub} | {rels[rel['r']]} | {obj})")
        #print(rel)
    gt.append('\n'.join(list(triples)))


data_pd = pd.DataFrame(zip(inst,gt),columns=['input', 'output'])
print(data_pd.head())
print(data_pd['input'][0])
print(data_pd['output'][0])
dataset = DatasetDict({"train": Dataset.from_pandas(data_pd)})
dataset.push_to_hub("UofA-LINGO/DocRED-train-noins", private=True)
#print(inst)
#print(gt)