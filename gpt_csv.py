from datasets import load_dataset
import pandas as pd

train_data = load_dataset("UofA-LINGO/wikipedia-summaries-gpt-triples", split="train")
train_df = pd.DataFrame(train_data)
train_df.to_csv("gpt_triples.csv")