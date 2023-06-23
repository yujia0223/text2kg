import pandas as pd  # Import pandas library
import os
import sys
from datasets import load_dataset
import fire
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pickle
#from utils import Iteratorize, Stream, Prompter
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from tqdm import tqdm

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(
    load_8bit: bool = False,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    csv_file: str = None,  # New argument for CSV file
):
    prompter = Prompter(prompt_template)

    # tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    
    model = LlamaForCausalLM.from_pretrained(
        "/home/taesiri/src/alpaca-lora/vicuna-7b--based-export-text-to-triplets-explanation-v3/",
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(Stream(callback_func=callback))
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)
    dfiles = {"train":"train.csv", "test":"test.csv"}
    dt = load_dataset("tsadler/text_to_triplets", data_files=dfiles)
    output = {}
    for i in tqdm(range(len(dt["test"]))):
        entry = dt["test"][i]
        output[i] = list(evaluate(entry["instruction"], entry["context"]))
        # print(output[i])
    with open("output-vicuna-7b-with-explanasion-correct.pickle", "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # generate a CSV
    dt = load_dataset("tsadler/text_to_triplets", data_files=dfiles)
    df = pd.DataFrame(dt["test"])
    df["gt"] = df["response"]
    df = df.drop(columns=["response"])
    df["model_output"] = [x[0] for x in output.values()]
    df.to_csv("vicuna-7b-with-explanasion-correct-test.csv", index=False)

    # dump df as pickle
    with open("vicuna-7b-with-explanasion-correct-df.pickle", "wb") as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire(main)
