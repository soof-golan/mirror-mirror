import timeit

import torch
from diffusers import StableDiffusionPipeline

repo = "IDKiro/sdxs-512-dreamshaper"
seed = 42
weight_type = torch.float16  # or float32

# Load model.
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    repo, torch_dtype=weight_type
)
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"
g = torch.Generator(device="cuda").manual_seed(seed)
prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
    [prompt],
    num_images_per_prompt=1,
    device=pipe.device,
    do_classifier_free_guidance=False,
)


# Ensure using 1 inference step and CFG set to 0.
def gen():
    image = pipe.__call__(
        # prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=1,
        guidance_scale=0,
        # generator=g,
        output_type="latent",
    ).images[0]


n = 100
t = timeit.timeit("gen()", globals=globals(), number=n)
print(f"Time taken for {n} runs: {t:.2f} seconds ({t / n:.2f} seconds per run)")
