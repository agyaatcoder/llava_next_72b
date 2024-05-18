import modal

stub = modal.Stub(image=modal.Image.debian_slim().pip_install("transformers==4.40.2", "huggingface_hub==0.22.2", "torch==2.3.0"))


volume = modal.Volume.from_name("hf-model-store", create_if_missing=True)


MODEL_NAME = "lmms-lab/llava-next-72b"
model_store_path = f"/vol/models/{MODEL_NAME}"


@stub.function(volumes={model_store_path: volume})
def seed_volume():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_store_path, exist_ok=True)

    snapshot_download(
        MODEL_NAME,
        local_dir=model_store_path,
        #ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        #token=os.environ["HF_TOKEN"],
        local_dir_use_symlinks=False
    )
    move_cache()
    print("Model downloaded successfully")

    volume.commit()




@stub.local_entrypoint()
def main(timeout: int = 10_000):
    # Write some images to a volume, for demonstration purposes.
    seed_volume.remote()
