from diffusers import StableDiffusionPipeline
import torch

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# Load LoRA checkpoint (your weights)
lora_weights = torch.load("./output/lora-2000.pt")

# Apply weights to matching layers in the UNet
unet = pipe.unet
with torch.no_grad():
    for name, param in unet.named_parameters():
        if name in lora_weights:
            print(f"Injecting weight into: {name}")
            param.copy_(lora_weights[name].to(param.device, dtype=param.dtype))

prompt = "photo of wife123 driving a car, DSLR photo, realistic, highly detailed"
negative_prompt = "cartoon, anime, blurry, ugly, disfigured, low quality, text, watermark"

for i in range(10):
    image = pipe(prompt, negative_prompt=negative_prompt,
                 guidance_scale=11.0).images[0]
    image.save(f"./output/image_{i}.png")
