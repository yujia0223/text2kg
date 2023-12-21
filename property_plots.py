import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

train_data = load_dataset("UofA-LINGO/webnlg-gpt-explanations", split="train")
test_data = load_dataset("UofA-LINGO/text_to_triplets", split="test")

cnt = 0
train_counts = dict()
train_props = []
test_counts = dict()
test_props = []
l = len("Therefore, here is the answer in the correct format:\n\n")
# Go through train set first
for entry in train_data['output']:
    # Get rid of description, and any extra triples generated before final output
    entry = entry[entry.find("Therefore, here is the answer in the correct format:\n\n")+l:]
    triples = entry.split('\',')
    for triple in triples:
        try:
            prop = triple.split('|')[1]
        except IndexError:
            print(cnt)
        if prop in train_counts:
            train_counts[prop] += 1
            train_props.append(prop)
        else:
            train_counts[prop] = 1
            train_props.append(prop)
    cnt+=1

cnt = 0
for entry in test_data['response']:
    triples = entry.split('\',')
    for triple in triples:
        try:
            prop = triple.split('|')[1]
        except IndexError:
            print(cnt)
        if prop in test_counts:
            test_counts[prop] += 1
            test_props.append(prop)
        else:
            test_counts[prop] = 1
            test_props.append(prop)
    cnt+=1

train_props.sort()
test_props.sort()
train_df = pd.DataFrame(train_props, columns=["property"])
test_df = pd.DataFrame(test_props, columns=["property"])

plt.figure(figsize=(40, 10))
hist_train = sns.histplot(train_df, x="property")
hist_train.set_xticklabels(hist_train.get_xticklabels(), rotation=90, horizontalalignment="right")
plt.show()

hist_test = sns.histplot(test_df, x="property")
hist_test.set_xticklabels(hist_test.get_xticklabels(), rotation=90, horizontalalignment="right")
plt.show()