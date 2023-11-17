import json
import fire

def load_json(filename):
    json_data = []
    with open(filename) as f:
        json_data = json.load(f)
    return json_data

def main(
    file: str = "",
):
    data = load_json(file)
    keys_scores = ['Ent_type', 'Partial', 'Exact', 'Strict']
    keys_metrics = ['Precision', 'Recall', 'F1']
    items = ['Correct', 'Incorrect', 'Partial', 'Missed', 'Spurious', 'Actual', 'Possible']
    for key in keys_scores:
        print(f"For {key}:\nPrecision: {data['Total_scores'][key][keys_metrics[0]]}     Recall: {data['Total_scores'][key][keys_metrics[1]]}     F1: {data['Total_scores'][key][keys_metrics[2]]}\n")
    for key in keys_scores:
        print(f"For {key}:")
        for item in items:
            print(f"{item}: {data['Total_scores'][key][item]}", end=" ")
        print()

if __name__ == "__main__":
    fire.Fire(main)