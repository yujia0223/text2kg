import json
import fire
import pandas as pd

def load_json(filename):
    json_data = []
    with open(filename, encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def main(
    in_file: str = "",
    out_file: str = ""
):
    if in_file == "" or out_file == "":
        print("Must enter a filename for --in_file and --out_file arguments.")
        exit(0)
    data = load_json(in_file)
    cands = []
    refs = []
    for i in range(len(data['cand'])):
        cands.append(" ".join(data['cand'][i]))
        refs.append(" ".join(data['ref'][i]))
    pd_data = pd.DataFrame(zip(cands, refs), columns=['Candidate', 'Reference'])
    print(pd_data)
    pd_data.to_csv(out_file, encoding='utf-8')


if __name__=="__main__":
    fire.Fire(main)