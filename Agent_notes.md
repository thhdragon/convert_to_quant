# Agent Notes

The current focus is getting the best quality out of int4 convrot w4a4 in comfy.
Unless we find a legitimate bug in the comfy kitchen implementation, our code
needs to integrate with it. That being said, we can use create custom comfy
nodes or modify kernels and load them from our custom node through the
comfy-kitchen backend.

location of comfy kitchen source code: .venv/lib/python3.13/site-packages/comfy_kitchen

Current PoC is flux2 klein 4B:
/tests/real_world_test_models/flux-2-klein-4b_bf16.safetensors

Next step is replacing random guassian calibration noise with actual flux2 klein latent steps collected from comfy.
Trying to standardize the calibration so using datasets from huggingface. Currently grabbed a random split of [UCSC-VLAA/GPT-Image-Edit-1.5M](https://ucsc-vlaa.github.io/GPT-Image-Edit/) (enhanced and standard prompts) and [ma-xu/fine-t2i](https://arxiv.org/abs/2602.09439) (ultraedit text/image pairs).
Flux2 Klein is distilled to 4 steps.
Currently we have ~1000 curated t2i prompts from fine-t2i but we only grabbed part 1 for PoC.
