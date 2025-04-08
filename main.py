from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverSinglestepScheduler
from transformers import CLIPVisionModelWithProjection
import torch
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controlnet_conditioning_scale = 1.0
ip_adapter_scale = 1.0

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    "DreamShaperXL.safetensors",
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")
pipe.set_ip_adapter_scale(ip_adapter_scale)

pipe = pipe.to(device)

pipe.enable_model_cpu_offload()

pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

init_image = Image.open("assets/init_image.png")
width, height = init_image.size

# create controlnet image canny
canny_image = Image.open("assets/line_art.png")

# reverse color
canny_image = Image.eval(canny_image, lambda x: 255 - x)

# Create line art if not available
# image = np.array(canny_image)
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)
# print(canny_image.size)
# canny_image.save("line_art.png")

# seed = 1337

light_cond = "morning"

output = pipe(
    prompt= f"{light_cond}, art, anime style, high quality",
    negative_prompt="human, animal, deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    image=canny_image,
    ip_adapter_image=init_image,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    # generator=torch.Generator(device=device).manual_seed(seed),
).images[0]

output.save("output.png")

