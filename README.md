# AnythingV3.0 Diffusion Model

Welcome to Anything V3 - a latent diffusion model for weebs. This model is intended to produce high-quality, highly detailed anime style with just a few prompts. Like other anime-style Stable Diffusion models, it also supports danbooru tags to generate images.

e.g. 1girl, white hair, golden eyes, beautiful eyes, detail, flower meadow, cumulonimbus clouds, lighting, detailed sky, garden 

Ran using Gradio WebUI

## Examples Compared with NovelAI

[Comparison 1](https://user-images.githubusercontent.com/1236582/201123592-e9018ce6-b446-4f25-87a5-5a8dacee05e8.png)

[Comparison 2](https://user-images.githubusercontent.com/1236582/201123915-ed41e734-f5fc-4040-947d-0aa8b9920e36.png)

[Comparison Table](https://user-images.githubusercontent.com/1236582/201127478-c2e9b844-db4d-4192-8524-5e24a05dda4c.jpg)


## Disclaimer
I made this collab for the purpose of running a few tests, and the queue times on hugging space is frustrating, and so using this google collab i was able to generate an image in a few seconds.

## Setup
Test it for yourself using this google collab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdeenSky/AnythingV3.0-WebUI/blob/main/AnythingV3.ipynb)

Make sure you've changed the runtime type in the collab, and select the hardware accelerator of "GPU" otherwise it will not work.
Run each cell in the notebook, until you get a public URL from the last cell.
<br><br>
<img src="https://i.imgur.com/DQoy6aS.png"></img>

Recommended Negative Prompt: 

lowres, bad anatomy, bad hands, text, error, missing fingers, bad feet, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name

## Credits
https://huggingface.co/spaces/akhaliq/anything-v3.0

https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/4516
