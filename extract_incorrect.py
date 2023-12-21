import json
import fire
from random import shuffle
import pandas as pd
import regex as re
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
    print(type(data))
    print(data.keys())
    # Just will have instruction text (done later), text + shuffed cand triples for input, ref triples or correct.
    output = []
    cnt = 0
    for i in range(len(data['id'])):
        output.append([])
        inp = ""
        inp += "Text: " + data['text'][i].strip() + "\n\n" + "Triples:\n"
        # Shuffle triples, convert to new (S,P,O) format.
        triples = data['cand'][i]
        try:
            assert(len(triples) > 0)
        except:
            print(i)
            continue
        shuffle(triples)
        tset = set()
        for triple in triples:
            tset.add(triple)
            triple = triple.split(' | ')
            assert(len(triple) == 3)
            inp += f"({triple[0]} | {triple[1]} | {triple[2]})\n"
        output[-1].append(inp.strip())
        if data['triple_score_sum'][i]['exact']['Incorrect'] or data['triple_score_sum'][i]['exact']['Missed'] or data['triple_score_sum'][i]['exact']['Spurious'] or len(triples) != len(tset):
            gt = ""
            # Shuffle triples, convert to new (S,P,O) format.
            triples = data['ref'][i]
            shuffle(triples)
            for triple in triples:
                triple = triple.split(' | ')
                assert(len(triple) == 3)
                gt += f"({triple[0]} | {triple[1]} | {triple[2]})\n"
            output[-1].append(gt.strip())
        else:
            cnt+=1
            output[-1].append("The generated triplets are already valid.")
    print(f"{cnt} fully correct, {len(data['id'])-cnt} with at least one incorrect.")

    data = pd.DataFrame(output, columns=["input", "output"])
    print(data.dropna())
    print(data)
    #print(data['input'][105])
    #print(re.findall(r"\((.*?)\)",data['input'][105]))
    #print(data['output'][105])
    #dataset = DatasetDict({"train": Dataset.from_pandas(data.dropna())})
    #dataset.push_to_hub("UofA-LINGO/text-to-triples-train-eval-no-instructions", private=True)
    return

if __name__ == "__main__":
    fire.Fire(main)