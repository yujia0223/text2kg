import fire
from datasets import load_dataset
import pandas as pd

def main(
    model_path: str = "",
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
    train_data = (
        train_val["train"].shuffle().map(generate_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_prompt)
    )
    print(type(train_data))
    print(type(val_data))

if __name__ == "__main__":
    fire.Fire(main)