import csv
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

def clean_line(inp):
    inp = inp.strip()
    inp = inp.replace('""', '"')
    if inp[-1] == '"':
        inp = inp[:len(inp)-1]
    if inp[0] == '"':
        inp = inp[1:]
    return inp

data = load_dataset("UofA-LINGO/ske-test")
panda_data = pd.DataFrame(data['test'])
print(panda_data)

panda_data.to_csv('test.csv')
"""
lines = []
with open('webnlg-test.csv', encoding='utf-8') as f:
    lines = f.readlines()

with open('webnlg-test.csv', 'w', encoding='utf-8') as f:
    for l in lines:
        if l.find(',"[') != -1:
            print(l[:l.find('"[')], file=f)
            print(l[l.find('"['):].strip(), file=f)
            #print(l, file=f)
        elif l.find(',[') != -1:
            print(l[:l.find('[')], file=f)
            print(l[l.find('['):].strip(), file=f)
            #print(l, file=f)
        else:
            print("error")
            print(l)
            break
        print("\n", file=f, end="")

lines = []
with open('webnlg-test-modified.csv', encoding='utf-8') as f:
    lines = f.readlines()
fixed = []
for i in range(0,len(lines),3):
    assert(lines[i+2] == '\n')
    fixed.append([clean_line(lines[i][:lines[i].find(',')]),
                  clean_line(lines[i][lines[i].find(',')+1:len(lines[i])-2]),
                  clean_line(lines[i+1])])
test = pd.DataFrame(fixed, columns=['index','input','output']).drop(columns=['index'])
lines = []
with open('webnlg-train-modified.csv', encoding='utf-8') as f:
    lines = f.readlines()
fixed = []
for i in range(0,len(lines),3):
    assert(lines[i+2] == '\n')
    fixed.append([clean_line(lines[i][:lines[i].find(',')]),
                  clean_line(lines[i][lines[i].find(',')+1:len(lines[i])-2]),
                  clean_line(lines[i+1])])
    if i == 9756*3:
        print(fixed[-1])
train = pd.DataFrame(fixed, columns=['index','input','output']).drop(columns=['index'])
dataset = DatasetDict({"train": Dataset.from_pandas(train), "test": Dataset.from_pandas(test)})
dataset.push_to_hub("UofA-LINGO/webnlg-cleaned-noins", private=True)
#print(type(test['output'][7648]))
#print(test['output'][7648].split('\', ')[1].strip("'"))
"""