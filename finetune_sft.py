"""
Resources Used:
https://huggingface.co/docs/trl/sft_trainer
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
https://github.com/huggingface/trl/pull/444#issue-1760952763
https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/trainer#transformers.TrainingArguments
"""

from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainerCallback, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
import fire
import os
import torch
import json

# From https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer
class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

# Attempt to avoid crashes due to cache not being cleared after saving / evaluation steps
class ClearGPUCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# Follows example at https://github.com/huggingface/trl/pull/444#issue-1760952763
def prompt_formatter(text):
    output_text = []
    template = os.environ.get("TEMPLATE")
    if template == None:
        template = "alpaca"
    template = template+".json"
    if not os.path.exists(template):
        print(f"Cannot read template: {template}")
        exit(0)
    template_str = ""
    with open(template) as f:
        template_str = json.load(f)
    for i in range(len(text['instruction'])):
        inst = text['instruction'][i]
        inp = text['input'][i]
        res = text['output'][i]
        out = template_str['prompt_input'].format(instruction=inst, input=inp)
        out = f"{out}{res}"
        output_text.append(out)
    return output_text

def train(
    model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # Lora params
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # Training params
    micro_batch_size: int = 4,
    batch_size: int = 32,
    log_steps: int = 10,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    val_set_size: int = 0,
    group_by_length: bool = False,
    # Wandb parameters
    wandb_project: str = "",
    prompt_template: str = "templates/alpaca",
):
    if model == "":
        print("Please provide model path, for example: /home2/tsadler/models/vicuna-7b")
        exit(0)

    if data_path == "":
        print("Please provide a dataset path, for example: UofA-LINGO/webnlg-cleaned-noref")
        exit(0)

    if output_dir == "":
        print("Please provide an output directory.")
        exit(0)

    use_wandb = True if len(wandb_project) > 0 else False
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
    
    os.environ["TEMPLATE"] = prompt_template
    token_path = model

    print(f"\nTraining model at: {model}")
    print(f"Using dataset at: {data_path}")
    print(f"Saving model at: {output_dir}")
    print(f"Using template at: {prompt_template}\n")
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map="auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dpp = world_size != 1
    
    if dpp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        model,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(token_path)
    tokenizer.padding_side = "right"
    tokenizer.padding_token_id = 0
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_data = None
    val_data = None
    if val_set_size == 0:
        train_data = load_dataset(data_path)['train'].shuffle()
    else:
        dat = load_dataset(data_path)['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = dat['train'].shuffle()
        val_data = dat['test'].shuffle()
    model = prepare_model_for_kbit_training(model)

    callbacks = [ClearGPUCallback(), PeftSavingCallback()]
    
    if not dpp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=prompt_formatter,
        peft_config=peft_config,
        callbacks=callbacks,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=log_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if dpp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
        )
    )

    trainer.train()
    # model.save_pretrained(output_dir)
    # Seems this is how to properly save the adapters
    trainer.save_model(output_dir)
    print("Model saved!")

fire.Fire(train)
