"""
Resources used:
https://huggingface.co/docs/trl/main/en/sft_trainer
https://twitter.com/AlphaSignalAI/status/1682815295893692416/photo/1

"""

from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from typing import List
import fire
import torch
import os
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# from https://github.com/tloen/alpaca-lora/issues/483, users ricksun2023 and maekawataiki
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # Fix by ricksun2023 for multi-GPU training
        try:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
        except Exception as e:
            print(f"Remove {pytorch_model_path} failed.")
        return control

class ClearGPUCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "",
    output_dir: str = "",
    val_split: str = "",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if base_model == "":
        print("Please specify a base model: --base_model=/home2/tsadler/models...")
        exit(0)
    if data_path == "":
        print("Please specify dataset: --data_path=UofA-LINGO/webnlg-reflections-updated-instructions")
        exit(0)
    if output_dir == "":
        print("Please specify directory to save output model: --output_dir=/home2/tsadler/models...")
        exit(0)
    if val_split == "":
        print("Please specify validaiton filename (csv, no extension): --val_split=val")
    gradient_accumulation_steps = batch_size // micro_batch_size
    train_data = load_dataset(data_path+'train.csv')
    val_data = load_dataset(data_path+val_split)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = "auto"
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16,
                                                device_map=device_map,)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        peft_config=config,
        torch_dtype=torch.float16,
        callbacks=[SavePeftModelCallback, ClearGPUCallback],
        max_seq_length=cutoff_len,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == '__main__':
    fire.Fire(train)