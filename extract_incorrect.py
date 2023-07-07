import json
import fire
from random import shuffle
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

def load_json(filename):
    json_data = []
    with open(filename) as f:
        json_data = json.load(f)
    return json_data

def main(
    details_file: str = "",
):
    data = load_json(details_file)
    # Just will have instruction text (done later), text + shuffed cand triples for input, ref triples or correct.
    output = []
    cnt = 0
    for i in range(len(data['id'])):
        output.append([])
        inp = ""
        inp += data['text'][i] + "\n"
        # Shuffle triples, convert to new (S,P,O) format.
        triples = data['cand'][i]
        shuffle(triples)
        for triple in triples:
            triple = triple.split(' | ')
            assert(len(triple) == 3)
            inp += f"({triple[0]}, {triple[1]}, {triple[2]})\n"
        output[-1].append(inp.strip())
        if data['triple_score_sum'][i]['exact']['Incorrect']:
            gt = ""
            # Shuffle triples, convert to new (S,P,O) format.
            triples = data['ref'][i]
            shuffle(triples)
            for triple in triples:
                triple = triple.split(' | ')
                assert(len(triple) == 3)
                gt += f"({triple[0]}, {triple[1]}, {triple[2]})\n"
            output[-1].append(gt.strip())
        else:
            cnt+=1
            output[-1].append("The generated triplets are already valid.")
    print(f"{cnt} fully correct, {len(data['id'])-cnt} with at least one incorrect.")

    data = pd.DataFrame(output, columns=["input", "output"])
    print(data.head())
    dataset = DatasetDict({"train": Dataset.from_pandas(data)})
    dataset.push_to_hub("UofA-LINGO/ttt-train-eval-no-instructions", private=True)
    return

if __name__ == "__main__":
    fire.Fire(main)