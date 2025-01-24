import os
import random
import torch
import yaml
from transformers import pipeline
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
from api import SimpleInpaintInfer, OutpaintInpaintInfer


# Load config
with open("api/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Setup segmentation pipeline once
seg_config = config["pipeline"]["segmentation_model"]
segmentation_models = [
    pipeline("image-segmentation", seg_config[i]) for i in range(len(seg_config))
]

# Setup StableDiffusion pipeline once
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config["pipeline"]["inpaint_model"],
        torch_dtype=torch.float16,
    ).to(config["pipeline"]["device"])
except:
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        config["pipeline"]["inpaint_model"],
        torch_dtype=torch.float16,
    ).to(config["pipeline"]["device"])

if config["pipeline"]["disable_safety_checker"]:
    pipe.safety_checker = None

# Load configuration values
lora_scale = config["general"]["lora_scale"]
negative_prompt = config["general"]["negative_prompt"]
not_included = config["general"]["not_included"]
not_included_body = config["general"]["not_included_body"]
not_included_face = config["general"]["not_included_face"]

# Image folder setup
folder_path = os.path.join(os.getcwd(), config["paths"]["input_folder"])
output_dir = os.path.join(os.getcwd(), config["paths"]["output_folder"])
os.makedirs(output_dir, exist_ok=True)

# Ensure prompts list exists
prompts = config["prompts"]
control_net_image = "examples/new/archive/1.jpg"
outpaint_dimensions = [(800, 640), (640, 800), (800, 800)]


# Processing Function
def process_image(i, file_name, prompt):
    try:
        image_path = os.path.join(folder_path, file_name)

        # Load image using custom Infer class
        inference_pipeline = OutpaintInpaintInfer(
            segmentation_model=segmentation_models, not_included=not_included, pipe=pipe
        )
        inference_pipeline.load_image(source=image_path, width=640)
        inference_pipeline.prepare_mask(dilation=20, visualize=True, invert_mask=False)

        # Generate inpainted image
        image = inference_pipeline(prompt, negative_prompt)

        # Save the resulting image
        result_path = os.path.join(
            output_dir, f"14th_{i}_{len(prompt)}_image_results.jpg"
        )
        image.save(result_path)

        print(f"Image saved: {result_path}")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    images = [
        img
        for img in os.listdir(folder_path)
        if img.endswith((".jpg", ".png", ".jpeg", ".webp"))
    ]
    for j, prompt in enumerate(prompts):
        print(prompt)
        for i, image in enumerate(images):
            process_image(i, image, prompt)
