import os
import sys

import fire
import gradio as gr
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

device = "cuda"


def main(
    model: str = "",
    tok: str = "",
    load_8bit: bool = False,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    prompter = Prompter(prompt_template)
    # tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    #tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    path = model
    if tok == "":
        tok = path
    tokenizer = LlamaTokenizer.from_pretrained(tok)
    model = LlamaForCausalLM.from_pretrained(
        path,
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
        num_beams=5,
        max_new_tokens=1024,
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
        eos_tokens = [tokenizer.eos_token_id]#, tokenizer.encode("<s>")[-1]]
        #eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("<s><|system|>")[-1], tokenizer.encode("<|system|>")[-1], tokenizer.encode("<s>[INST]")[-1], tokenizer.encode("<<SYS>>")[-1], tokenizer.encode(")<")[-1], tokenizer.encode("<s>")[-1]]
        #eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("<s>"), tokenizer.encode("<|system|>"), tokenizer.encode("<s>[INST]"), tokenizer.encode("<<SYS>>")]
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_tokens,
            "pad_token_id": 0,
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
                print(kwargs)
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)
                    print("Output:")
                    print(output)
                    print("\nDecoded Output:")
                    print(decoded_output)
                    if output[-1] in eos_tokens:
                        print("IN BREAK")
                        print(decoded_output)
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
                eos_token_id=eos_tokens,
                pad_token_id=0,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
                value="Generate triplets from the text below:",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=8, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=1024, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output", value=True),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title=f"Text to Triplets - using \'{path}\'",
        description="Generate Triplets using custom Alpaca-LoRA model",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
