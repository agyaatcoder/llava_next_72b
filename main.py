import modal
from uuid import uuid4

MODEL_PATH = "lmms-lab/llava-next-72b"
GPU_CONFIG = modal.gpu.A100(count=4, memory = 80)

volume = modal.Volume.from_name("hf-model-store", create_if_missing=True)

model_store_path = f"/vol/models/{MODEL_PATH}"
MINUTES = 60 

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "cmake")
    .pip_install(
        "wheel==0.43.0",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "transformers==4.40.2",
        "vllm==0.4.2",
        "timm==0.9.12",
        "Pillow==10.3.0",
        "huggingface_hub==0.22.2",
        "requests==2.31.0",
        "einops",
        "accelerate",
        # force_build=True
    )
    .run_commands("pip install git+https://github.com/agyaatcoder/LLaVA-NeXT.git")
)

stub = modal.Stub("my-app")

with vllm_image.imports():
    from PIL import Image

@stub.cls(
    timeout=20 * MINUTES,
    container_idle_timeout=150,
    allow_concurrent_inputs=10,
    image=vllm_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={model_store_path: volume},
    gpu = GPU_CONFIG
)
class Model:
    @modal.enter()
    def start_engine(self):
        from llava.model.builder import load_pretrained_model
        import torch
        from pathlib import Path
        
        model_path = Path('/vol/models/lmms-lab/llava-next-72b')

        model_name = "llava_qwen"
        self.device = "cuda"
        self.device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, 
                                                                                                  model_name, attn_implementation= None, 
                                                                                                  device_map=self.device_map) # Add any other thing you want to pass in llava_model_args

        self.model.eval()
        self.model.tie_weights()

 
    @modal.method()
    def generate(self, url):
        import requests
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN # DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates
        import torch
        #import PIL
        from PIL import Image
        import copy

        print("Generating...")
        # #url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        # response = requests.get(url, stream=True)
        # content_type = response.headers.get('content-type')
        # if 'image' not in content_type:
        #      raise ValueError(f"Invalid content type: {content_type}")
        
        # try:
        #     image = PIL.Image.open(requests.get(url, stream=True).raw)
        # except PIL.UnidentifiedImageError as e:
        #     print(f"Error opening image: {e}")
#image = Image.open(response.raw)
        image_filename = url.split("/")[-1]
        image_path = f"/tmp/{uuid4()}-{image_filename}"
        response = requests.get(url)
        response.raise_for_status()
        with open(image_path, "wb") as file:
            file.write(response.content)

        image = Image.open(image_path)

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        print(image_tensor)

        conv_template = "qwen_1_5"  #"llava_llama_3" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        return text_outputs


@stub.local_entrypoint()
def main():
    url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    model = Model()

    result = model.generate.remote(url)
    print(result)

