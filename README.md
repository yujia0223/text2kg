# Lingo-Scripts

This is the repository containing the majority of the files used for training and evaluating models, storing their results and other "helper" files. The details of how to use these files is provided in this README, and further details are provided in the README files contained in each individual folder.

## Training

Model training is carried out with the finetune.py (from alpaca-lora) and finetune_sft.py scripts as well as autotrain. Please note sft_train is an older version of finetune_sft, just kept for documentation purposes that should not be used to train models.

### finetune.py

For using finetune.py from alpaca-lora, the version from this repository should be used to replace the file after cloning that repository. This is because there are modifications to the version in this repository for saving models and clearing the GPU cache that improve performance and ensure models are properly saved. Ensure the `torch2-tune` environment is active, as explained in the env_files folder. An example command for running the script is below:

`python /home/tyler/alpaca-lora/finetune.py --base_model=/mnt/tyler/models/base_models/orca_mini_v3_7b/ --data_path=UofA-LINGO/FewRel-train --output_dir=/mnt/tyler/models/orca-mini-3-7b-fewrel/ --num_epochs=1 --cutoff_len=1024 --group_by_length --lora_target_modules=[q_proj,k_proj,v_proj,o_proj] --lora_r=16 --micro_batch_size=4 --batch_size=128 --val_set_size=0`

The relevant arguments, as seen above are as follows:
- --base_model; provides the path to the model that will be fine-tuned, either on the system or from HuggingFace.
- --data_path; provides the path to the data to be used in training, either on the system or from HuggingFace.
- --output_dir; the path to export the adapter weights for the completed models to.
- --num_epochs; the number of epochs to train the model on.
- --cutoff_len; the number of tokens the model is allowed to generate in its reponse.
- --group_by_len; speeds up training, but may be responsible for strange loss curves.
- --lora_target_modules; the lora modules supplied to the LoraConfig. We trained all our models with the list in the example.
- --lora_r; r value supplied to LoraConfig.
- --lora_dropout; the dropout rate for LoraConfig. Left at default (0.05) for our models.
- --lora_alpha; the alpha supplied to LoraConfig. Left at default (16) for our models.
- --micro_batch_size; used to set the per_device_train_batch_size and gradient accumulation steps (batch_size // micro_batch_size).
- --batch_size; batch size for training. As explained above, also sets the gradient accumulation steps.
- --val_set_size; the size of datapoints to be included in the validation set. Size of 0 skips validation.
- --learning_rate; defaults to 3e-4, which worked well for our models so was not set in the above example.
- --prompt_template_name; name of the prompt template to use for formatting data given to the model. See the templates directory for more information.

There are other command line arguments available, but were left at their default values when training for our models. Other parts of the file that may be relevant to change are the eval_steps and save_steps, which are not provided as arguments but may be changed directly in the file (lines 278 and 279 in the version in this repository). Eval_steps controls how many training steps occur before the validation data is run. Evaluation can take some time, so increasing this number can speed up training but will give a less clear evaluation loss curve. Save_steps are the number of steps before a checkpoint is saved during training. We did not make use of checkpoints, but they do allow for training to be restarted from a given point. This number should be a multiple of eval_steps.

To train on multiple GPUs, replace "python" in the example command with:
`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234`. CUDA_VISIBLE_DEVICES should be set according to the number of GPUs on the machine, and which ones are to be used for training. The available GPUs can be checked with either `nvitop` or `nvidia-smi`. A full example of this command, used on the lambda machine is as follows:

`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune.py --base_model='/home2/tsadler/models/Llama-2-7b-hf' --data_path='UofA-LINGO/webnlg-combined-with-reflections' --output_dir='/home2/tsadler/models/Llama-2-7b-combined-with-reflection' --num_epochs=1 --cutoff_len=1024 --group_by_length --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --lora_r=16 --micro_batch_size=4 --batch_size=64  --val_set_size=8000 --prompt_template_name=llama1`

### finetune_sft.py

This script is very similar to finetune.py from alpaca-lora but makes use of the SFTTrainer instead of the default HuggingFace Trainer. Command line arguments are similar to finetune.py, but many of the ones that typically went unused have been removed. 