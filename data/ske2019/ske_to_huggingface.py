from datasets import DatasetDict, Dataset
import pandas as pd

order = ["instruction", "input", "output"]
train = pd.read_csv("new_train.csv")
train = train.rename(columns={"context(string)": "input", "response(string)": "output", "instruction(string)": "instruction"})
train = train.reindex(columns=order)
val = pd.read_csv("new_valid.csv")
val = val.rename(columns={"context(string)": "input", "response(string)": "output", "instruction(string)": "instruction"})
val = val.reindex(columns=order)
test = pd.read_csv("new_test.csv")
test = test.rename(columns={"context(string)": "input", "response(string)": "output", "instruction(string)": "instruction"})
test = test.reindex(columns=order)
print("Training:")
print(train.head())
print("Valid:")
print(val.head())
print("Test:")
print(test.head())
dataset = DatasetDict({"train": Dataset.from_pandas(train), "valid": Dataset.from_pandas(val), "test": Dataset.from_pandas(test)})
dataset.push_to_hub("UofA-LINGO/ske2019", private=True)
