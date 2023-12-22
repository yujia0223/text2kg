# Lingo-Scripts

This is the repository containing the majority of the files used for training and evaluating models, storing their results and other "helper" files. The details of how to use these files is provided in this README, and further details are provided in the README files contained in each individual folder.

Please note, the finetune.py, finetune_sft.py and benchmark_and_evaluate.py files must be run from the base lingo-scripts directory, as they make use of the files in the utils and templates directories. Finetune.py may alternatively be run inside the alpaca-lora base directory, after cloning the repository from GitHub.

## Training

Model training is carried out with the finetune.py (from alpaca-lora) and finetune_sft.py scripts as well as autotrain. Please note sft_train is an older version of finetune_sft, just kept for documentation purposes that should not be used to train models. The training command used for each model can be found on its associated HuggingFace page.

Please note, the .sh files should not be used as is for training new models. These were used to essentially "queue" many training jobs at once, to ensure as soon as one model finished the next one would start right away. If this approach is being used, write your new training commands into a .sh file then run it. A similar approach was also used for evaluation, and can also be applied in the same way by writing new evaluation commands into a .sh file first then running it later.

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
- --prompt_template_name; name of the prompt template to use for formatting data given to the model. See the templates directory for more information. Defaults to alpaca.
- --wandb_project; name of the wandb project to save logging output to.
- --wandb_run_name; name of the run in wandb.

There are other command line arguments available, but were left at their default values when training for our models. Other parts of the file that may be relevant to change are the eval_steps and save_steps, which are not provided as arguments but may be changed directly in the file (lines 278 and 279 in the version in this repository). Eval_steps controls how many training steps occur before the validation data is run. Evaluation can take some time, so increasing this number can speed up training but will give a less clear evaluation loss curve. Save_steps are the number of steps before a checkpoint is saved during training. We did not make use of checkpoints, but they do allow for training to be restarted from a given point. This number should be a multiple of eval_steps.

To train on multiple GPUs, replace "python" in the example command with:
`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234`. CUDA_VISIBLE_DEVICES should be set according to the number of GPUs on the machine, and which ones are to be used for training. The available GPUs can be checked with either `nvitop` or `nvidia-smi`. A full example of this command, used on the lambda machine is as follows:

`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune.py --base_model='/home2/tsadler/models/Llama-2-7b-hf' --data_path='UofA-LINGO/webnlg-combined-with-reflections' --output_dir='/home2/tsadler/models/Llama-2-7b-combined-with-reflection' --num_epochs=1 --cutoff_len=1024 --group_by_length --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --lora_r=16 --micro_batch_size=4 --batch_size=64  --val_set_size=8000 --prompt_template_name=llama1`

### finetune_sft.py

This script is very similar to finetune.py from alpaca-lora but makes use of the SFTTrainer instead of the default HuggingFace Trainer. This script should be used with `torch2-auto` or `torch2-new` as the active environment. Command line arguments are similar to finetune.py, but many of the ones that typically went unused have been removed. The example commands used with finetune.py can also be used here, with some changes to command line arguments, which are as follows:

- --model; same as base_model from above. The model to be fine-tuned.
- --data_path; same as above, the path to the dataset to use for training either on the machine or HuggingFace.
- --output_dir; same as above, the location to save the adapter weights to.
- --lora_r; r value supplied to LoraConfig.
- --lora_dropout; the dropout rate for LoraConfig. Left at default (0.05) for our models.
- --lora_alpha; the alpha supplied to LoraConfig. Left at default (16) for our models.
--micro_batch_size; used to set the per_device_train_batch_size and gradient accumulation steps (batch_size // micro_batch_size).
- --batch_size; batch size for training. As explained above, also sets the gradient accumulation steps.
- --val_set_size; the size of datapoints to be included in the validation set. Size of 0 skips validation.
- --learning_rate; defaults to 3e-4, which worked well for our models so was not set in the above example.
- --log_steps; the number of training steps taken for logging the training loss. Defaults to 10, no noticable effect on performance.
- --num_epochs; the number of epochs to train the model on.
- --group_by_len; speeds up training, but may be responsible for strange loss curves.
- --wandb_project; name of the wandb project to save logging output to.
- --prompt_template; name of the prompt template to use for formatting data given to the model. See the templates directory for more information. Defaults to alpaca.

Again, eval_steps and save_steps may be changed in the code. These can be found on lines 164 and 165. An example of a training command for use on the lambda machines is below:

`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune_sft.py --model='/home2/tsadler/models/vicuna-7b' --data_path='UofA-LINGO/webnlg-combined-with-reflections' --output_dir='/home2/tsadler/models/vicuna-7b-sft' --num_epochs=1 --group_by_length --lora_r=16 --micro_batch_size=4 --batch_size=64  --val_set_size=8000`

### autotrain

For using autotrain, the `torch2-auto` or `torch2-new` environment should be active. It is very similar to running one of the above scripts, just with slight differences in formatting the command line arguments. Additionally, for training on multiple GPUs, the GPUs available for training should be set as environment variables, that is before training if you wanted to use both GPUs set the CUDA_VISIBLE_DEVICES environment variable with:

`export CUDA_VISIBLE_DEVICES=0,1`

After this, run autotrain. An example command run in the `torch2-new` environment is as follows:

`autotrain llm --train --model /home2/tsadler/models/base_models/Mistral-7B-Instruct-v0.1/ --data_path /home2/tsadler/data/webnlg-rf-mistral --valid_split val --project_name /home2/tsadler/models/mistral-7b-at-sft --use_peft --train_batch_size 8 --num_train_epochs 1 --learning_rate 3e-4 --use_int4 --trainer sft --lora_r 16 --text_column text --target_modules q_proj,k_proj,o_proj,v_proj`

The full list of command line arguments can be found with `autotrain llm --help`. Relevant arguments are as follows:
- --train; tells autotrain to train the model.
- --model; the path to the model to be trained, either on the computer or on HuggingFace.
- --data_path; the path to the data to use for training. For our purposes, should be a directory containing data formatted with one of the prep_data files. More details on this in the data directory.
- --valid_split; the file name of the validation data file, without any file extension.
- --project_name; the output directory for the adapter weights of the fine-tuned model.
- --use_peft; a flag to tell autotrain to use PEFT. Should be included.
- --train_batch_size; the training batch size. In general, autotrain seems to use more RAM on the GPU than the finetune scripts, so this number has to be smaller to compensate.
- --num_train_epochs; the number of training epochs to complete.
- --learning_rate; the learning rate for training
- --use_intX; replace X with 4 or 8. In general, due to increased memory usage use_int4 had to be used.
- --trainer; sets the type of trainer to use, sft seemed to be the best available for autotrain.
- --lora_r; the r value for LoRA.
- --text_column; the name of the column used to get the training data text from. If data is formatted with prep_data, this column will be named text, so the argument should be used as demonstrated in the example.
- --target_modules; the target modules for LoRA. **NOTE:** This argument can only be used in the `torch2-new` environment.

### Applying Adapter Weights

To finish the training process, the adapter weights have to be applied back to the base models and saved, so that they can be later used for inference. The export_hf_checkpoint.py file does this for us. The following steps should be taken to apply the adapter weights to a model trained with one of the above methods:

1. Navigate to the directory containing the adapter weights.
2. Set the BASE_MODEL environment variable to the model that was fine-tuned. For example, if we finetuned llama-2-13b, we could run: `export BASE_MODEL=/home/tsadler/models/base_models/llama-2-13b`
3. Set the PEFT_MODEL environment variable to the directory containing the adapter weights. This should be the directory you are currently in. Again, use `export PEFT_MODEL=...`
4. Run the export_hf_checkpoint.py file while still in the directory with the adapter weights. It will apply the weights, and save the output to a new directory called `hf_ckpt`.
5. For benchmarking, typically you will have to supply separate paths for the model and the tokenizer, as only autotrain saves a copy of the tokenizer in the output directory. Even in this case, make sure to supply separate paths, as the path to the model must include the `/hf_ckpt` at the end, whereas the path to the tokenizer will not.


## Benchmark and Evaluate

There are a few benchmarking files left here but only three should be used, and the rest have just been left for documentation purposes incase versions are helpful.

### benchmark_and_evaluate.py

This file is the main one that should be used for evaluation of the models. It will run the model on all examples in the specified test data, then evaluate its performance. Additionaly, it can start from saved output and just evaluate on a given test set. This file was used to get the results for all models we trained. The command line arguments to use with this file are as follows:

- --model_path; the path to the model that is to be benchmarked.
- --tok; the path to the tokenizer to use in benchmarking.
- --prompt_template; the name of the prompt template to use during benchmarking. Defaults to alpaca.
- --max_tokens; the maximum number of tokens the model can generate during benchmarking.
- --dump; the output pickle file to "dump" the raw model output into after benchmarking. Make sure to set this, as the benchmarking will not need to run again if anything goes wrong during evaluation.
- --load_8bit; a flag to load the model in 8bit mode for benchmarking. Sometimes needs to be used for larger (13b+) models, but should not ever need to be used for 7b models.
- --test; the path to the test dataset to use. Defaults to UofA-LINGO/text_to_triplets_new_ins. Can be a path on the machine or from HuggingFace.
- --error; the file to output any error messages encountered during benchmarking. Should not be an issue with current models, but was needed when models required BOS tokens to be set as EOS tokens as well.
- --output_path; the path to output the scored metrics to. This file will only contain scores, no text or triples.
- --output_details_path; the path to output the verbose output file to. This file will contain scores, input text, reference triples and candidate triples.
- --pickle; the path to a pickle file containing previously saved output. **NOTE:** If this argument is used, only the evaluation portion of the file will run on the saved output. Thus, if this argument is used model_path, tok, prompt_template, max_tokens, dump and load_8bit do not need to be set.

Below is an example of using the file to benchmark and evaluate the results of a fine-tuned model:

`python benchmark_and_evaluate.py --model_path=/home/tsadler/models/orca-mini-3-at-sft/hf_ckpt --tok=/home/tsadler/models/orca-mini-3-at-sft --max_tokens=1024 --dump=/home/tsadler/lingo-scripts/raw_outputs/orca-mini-3-at-sft.pickle --output_path=/home/tsadler/lingo-scripts/results/orca-mini-3-at-sft.json --output_details_path=/home/tsadler/lingo-scripts/results/orca-mini-3-at-sft-det.json`

Another example is below, this time running just the evaluation portion of the code on the results that would have been generated by the above example:

`python benchmark_and_evaluate.py --pickle=/home/tsadler/lingo-scripts/raw_outputs/orca-mini-3-at-sft.pickle --output_path=/home/tsadler/lingo-scripts/results/orca-mini-3-at-sft.json --output_details_path=/home/tsadler/lingo-scripts/results/orca-mini-3-at-sft-det.json`

If using the pickle argument, like in the example above, there should be no GPU usage by the script as it will never load any models. If benchmarking a model, one GPU will usually be fully utilized on the lambda machines. As they have two, it is often a good idea to fine-tune all models first, then benchmark half on one GPU and half on the other to speed up the overall process. To ensure only one GPU is used, set the CUDA_VISIBLE_DEVICES environment variable to be equal to the GPU that is to be used. For example, if you want to benchmark a model on GPU 1, use: `export CUDA_VISIBLE_DEVICES=1`. Two GPUs can be helpful to use, as they are just enough to benchmark a 13b model, although outside of this case one should be enough.

### benchmark_gpt.py

This file was specifically used for benchmarking results from the most recent run of gpt-4 on the full webnlg test set, as it required more preprocessing than is normally applied to models. For running this on other instances of GPT output, it may need some modifications. This file should not be used to benchmark or evaluate any other models. Command line arguments are shared with benchmark_and_evaluate.py so will not be repeated here.

### benchmark_for_training.py

This file is for generating benchmarked output on just the text to triples task of the webnlg dataset. Note that this version uses very old instructions, and the version of dataset should be modified if output such as this is gathered again. The output we got from this was used to create the "reflection" task in future versions of our webnlg dataset. Relevant command line arguments are as follows:

- --model_path; path to the model to use for generation of triples.
- --tok; path to the tokenizer used by the model, if it is different from the model path.
- --dump; the path to the pickle file used to "dump" the raw model output into. For this file, this is where you can get the output of the model on the training data.

## Other Useful Files

There are many files not covered above, although only some are useful as others are older files or just used for a very specific purpose. Some of the more useful files are explained below:

### get_eval_results.py

This file is used to extract the result scores from the JSON files generated by benchmark_and_evaluate.py. Some further details about this are in the results directory. Usage: `python get_eval_results.py --file=PATH`

### extract_det.py

This file is used to extract the reference and candidate triples from a detailed results JSON file generated by benchmark_and_evaluate.py. Some futher details about this are in the results directory. Usage: `python extract_det.py --det_file=PATH`

### det_to_csv.py

This file is used to turn one of the detailed output JSON files from benchmark_and_evaluate.py into a CSV file, as in some cases like with the SKE data the JSON file will not display the proper UTF-8 characters, thus the output is not human readable. CSV files don't seem to have this problem, so this helper script can convert the JSON files into CSV format. Usage: `python det_to_csv.py --in_file=DETAILS_FILE_PATH --out_file=OUTPUT_CSV_FILE_PATH`

### upload_model.py

Used for uploading models, or datasets, to HuggingFace. This file will require a bit of modification to be used, as all paths are hardcoded to tsadler paths. After updating the paths to point to wherever you store models, the file can be used as follows: `python upload_model.py --model=MODEL_NAME`. Note that MODEL_NAME should be the same on the machine and HuggingFace. For uploading a dataset, just modifiy the path to point to the directory containing the data you want to upload, and change repo_type to be equal to "dataset".

### chatgpt.py

Essentailly a python file version of the ChatGPT_API_Wrapper.ipynb Jupyter notebook file. This allows you to interact with the ChatGPT website though a Python file and have the results printed to the command line. Do not use this file to make automated requests, as it may result in the account being banned. It is left here for documentation purposes, but currently any lines that make requests are commented out.

### tsv_to_pickle.py

This file was used to convert the TSV/CSV files used to store ChatGPT output into pickle files so they could more easily be used by benchmark_and_evaluate.py. There are no input arguments as script is only really useful for the described purpose. Depending on the format of the output, more preprocessing is done within this file before ouputting the final pickle file.