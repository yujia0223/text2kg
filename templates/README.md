# Templates

This is the templates directory from the [alpaca-lora](https://github.com/tloen/alpaca-lora) repository, with some extra templates we have used. The description of our templates has been added alongside the existing descriptions below.

# Prompt templates

This directory contains template styles for the prompts used to finetune LoRA models.

## Format

A template is described via a JSON file with the following keys:

- `prompt_input`: The template to use when input is not None. Uses `{instruction}` and `{input}` placeholders.
- `prompt_no_input`: The template to use when input is None. Uses `{instruction}` placeholders.
- `description`: A short description of the template, with possible use cases.
- `response_split`: The text to use as separator when cutting real response from the model output.

No `{response}` placeholder was used, since the response is always the last element of the template and is just to be concatenated to the rest.

## Example template

The default template, used unless otherwise specified, is `alpaca.json`

```json
{
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

```

## Current templates

### alpaca

Default template used for generic LoRA fine tunes so far.

### alpaca_legacy

Legacy template used by the original alpaca repo, with no `\n` after the response field. Kept for reference and experiments.

### alpaca_short

A trimmed down alpaca template which seems to perform just as well and spare some tokens. Models created with the default template seem to be queryable by the short tempalte as well. More experiments are welcome.

### vigogne

The default alpaca template, translated to french. This template was used to train the "Vigogne" LoRA and is to be used to query it, or for extra fine tuning.

### llama1

The main template we use for training llama-2 models. The template was found at [OpenAssistant/llama2-13b-orca-8k-3319](https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-3319) and worked well for fine-tuning. For fine-tuning with only a few samples of data, llama2 should be used as it is the one utilized by Meta.

### llama2

The Meta template for llama-2 models. Used for benchmarking base models or fine-tuning them. Similar performance to llama1.

### orca_mini

A template for orca_mini models, although it was found to work worse than the alpaca template. As such, this template was mostly unused but is left here for documentation purposes. Please use alpaca for training orca_mini models.

### llongorca

The template used for tuning and benchmarking the LlongOrca models, found directly at the [model page](https://https://huggingface.co/Open-Orca/LlongOrca-7B-16k).

### mistral

The template used for training the Mistral models, found at their [model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).