import timeit

import torch
from diffusers import StableDiffusionPipeline

repo = "IDKiro/sdxs-512-dreamshaper"
seed = 42
weight_type = torch.float16  # or float32

# Load model.
pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"


# Ensure using 1 inference step and CFG set to 0.
def gen():
    image = pipe(
        prompt,
        num_inference_steps=1,
        guidance_scale=0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]

    image.save("output.png")


t = timeit.timeit("gen()", globals=globals(), number=10)
print(f"Time taken for 10 runs: {t:.2f} seconds")
