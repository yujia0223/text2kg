import fire
from datasets import load_dataset
import pandas as pd

def main(
    data_path: str = "",
    val_set_size: int = 0
):
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    def generate_prompt(data_point):
        res = prompt.format(instruction=data_point["instruction"], input=data_point["input"])
        res = f"{res}{data_point['output']}"
        return res

    data = load_dataset(data_path)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = []
    for data in train_val['train'].shuffle():
        train_data.append(generate_prompt(data))
    val_data = []
    for data in train_val['test'].shuffle():
        val_data.append(generate_prompt(data))
    train_data = pd.DataFrame(train_data, columns=['text'])
    val_data = pd.DataFrame(val_data, columns=['text'])
    print(train_data.head())
    print(val_data.head())


if __name__ == "__main__":
    fire.Fire(main)