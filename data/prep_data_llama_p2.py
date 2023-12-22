import fire
from datasets import load_dataset
import pandas as pd

def main(
    data_path: str = "",
    val_set_size: int = 0
):
    system_prompt = "You are an AI assistant who is an expert in knowledge graphs. You will be given an instruction and text. Generate a response to appropriately complete the instruction's request."
    prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}\n\n{input} [/INST]\n"
    def generate_prompt(data_point):
        res = prompt.format(system_prompt = system_prompt, instruction=data_point["instruction"], input=data_point["input"])
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
    print(train_data['text'][0])
    print(val_data['text'][0])
    train_data.to_csv("train.csv", index=False)
    val_data.to_csv("val.csv", index=False)

if __name__ == "__main__":
    fire.Fire(main)