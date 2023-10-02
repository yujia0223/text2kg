from huggingface_hub import HfApi
import fire

def main(
    model: str = "vicuna-7b"
):
    api = HfApi()
    api.upload_folder(
        #folder_path="/home2/tsadler/models/"+model,
        folder_path="/home/tsadler/lingo-scripts/raw_outputs",
        #repo_id="UofA-LINGO/"+model,
        repo_id="UofA-LINGO/model-benchmark-outputs",
        repo_type="dataset",
        ignore_patterns="**pytorch_model**",
    )

fire.Fire(main)


