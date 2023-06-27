from datasets import load_dataset

dt = load_dataset("UofA-LINGO/text_to_triplets")

dt2 = load_dataset("UofA-LINGO/webnlg-triplets-explanation-v1")
print(dt2["test"][0]['context'] == dt["test"][0]['context'])

for i in range(dt['test'].num_rows):
    if dt["test"][i]['context'] != dt2["test"][i]["context"]:
        print("Not equal")
        break
print("Equal")