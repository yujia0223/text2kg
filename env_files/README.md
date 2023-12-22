## Environment Setup

To setup one of the environments used for training and benchmarking models, use the environment files here. Models were benchmarked in the same environment they were trained in.

### Torch2-tune
This environment is for use with the alpaca-lora finetune.py script. It is not compatible with any version of autotrain. To create the environment, run the following:

```conda env create -f torch2-tune.yml```

**NOTE:** This environment requires peft==0.3.0.dev0, which is not automatically downloaded using the file. After the environment is created, activate it with ```conda activate torch2-tune``` then run the following:

```pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08```

### Torch2-auto
This environment is for use training models with autotrain or the finetune_sft.py script. This version of autotrain was used for the bulk of the models we trained using this method. This environment is not compatible with Mistral models. Install the environment with:

```conda env create -f torch2-auto.yml```

### Torch2-new
This environment is a newer version of the autotrain environment that allows us to train and benchmark Mistral models and any models using its architecture. This environment was used for training Mistral and SOLAR so far. Install with:

```conda env create -f torch2-new.yml```

**NOTE:** There may be an issue installing transformers, as there version is transformers==4.34.0.dev0. The transformers version can be safely updated for future use, as this version had to be installed from source as we started using this environment the week the Mistral models were released, which required source installation of transformers.