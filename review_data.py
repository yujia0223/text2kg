from datasets import load_dataset
import pandas as pd

data = pd.DataFrame(load_dataset('UofA-LINGO/ske-train')['train'])
print(data['output'][0])