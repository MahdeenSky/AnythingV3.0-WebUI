from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image
import utils
import datetime
import time
import psutil

start_time = time.time()
is_colab = utils.is_google_colab()

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

models = [
   Model("anything v3", "Linaqruf/anything-v3.0", "anything v3 style"),
  ]
    #  Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
    #  Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
    #  Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
    #  Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy ")
     #Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
     #Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
     #Model("Robo Diffusion", "nousr/robo-diffusion", ""),
     
scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

custom_model = None
if is_colab:
  models.insert(0, Model("Custom model"))
  custom_model = models[0]

last_mode = "txt2img"
current_model = models[1] if is_colab else models[0]
current_model_path = current_model.path

if is_colab:
  pipe = StableDiffusionPipeline.from_pretrained(current_model.path, torch_dtype=torch.float16, scheduler=scheduler, safety_checker=lambda images, clip_input: (images, False))

else: # download all models
  print(f"{datetime.datetime.now()} Downloading vae...")
  vae = AutoencoderKL.from_pretrained(current_model.path, subfolder="vae", torch_dtype=torch.float16)
  for model in models:
    try:
        print(f"{datetime.datetime.now()} Downloading {model.name} model...")
        unet = UNet2DConditionModel.from_pretrained(model.path, subfolder="unet", torch_dtype=torch.float16)
        model.pipe_t2i = StableDiffusionPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
        model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
    except Exception as e:
        print(f"{datetime.datetime.now()} Failed to load model " + model.name + ": " + str(e))
        models.remove(model)
  pipe = models[0].pipe_t2i
  
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def on_model_change(model_name):
  
  prefix = "Enter prompt. \"" + next((m.prefix for m in models if m.name == model_name), None) + "\" is prefixed automatically" if model_name != models[0].name else "Don't forget to use the custom model prefix in the prompt!"

  return gr.update(visible = model_name == models[0].name), gr.update(placeholder=prefix)

def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  print(psutil.virtual_memory()) # print memory usage

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None

  try:
    if img is not None:
      return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None
    else:
      return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator), None
  except Exception as e:
    return None, error_str(e)

def txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} txt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16, scheduler=scheduler, safety_checker=lambda images, clip_input: (images, False))
        else:
          pipe = pipe.to("cpu")
          pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt  
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      # num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return replace_nsfw_images(result)

def img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16, scheduler=scheduler, safety_checker=lambda images, clip_input: (images, False))
        else:
          pipe = pipe.to("cpu")
          pipe = current_model.pipe_i2i
        
        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        # num_images_per_prompt=n_images,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)
        
    return replace_nsfw_images(result)

def replace_nsfw_images(results):

    if is_colab:
      return results.images[0]
      
    for i in range(len(results.images)):
      if results.nsfw_content_detected[i]:
        results.images[i] = Image.open("nsfw.png")
    return results.images[0]

css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Anything V3</h1>
              </div>
              <p>
               Demo for Anything V3
              </p>
              <p>You can skip the queue by duplicating this space: <a style="display:inline-block" href="https://huggingface.co/spaces/akhaliq/anything-v3.0?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>       </p>
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
              with gr.Box(visible=False) as custom_model_group:
                custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", interactive=True)
                gr.HTML("<div><font size='2'>Custom models have to be downloaded first, so give it some time.</font></div>")
              
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="Enter prompt. Style applied automatically").style(container=False)
                generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))


              image_out = gr.Image(height=512)
              # gallery = gr.Gallery(
              #     label="Generated images", show_label=False, elem_id="gallery"
              # ).style(grid=[1], height="auto")
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            with gr.Group():
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

              # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                steps = gr.Slider(label="Steps", value=25, minimum=2, maximum=75, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

              seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

          with gr.Tab("Image to image"):
              with gr.Group():
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    if is_colab:
      model_name.change(on_model_change, inputs=model_name, outputs=[custom_model_group, prompt], queue=False)
      custom_model_path.change(custom_model_changed, inputs=custom_model_path, outputs=None)
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    ex = gr.Examples([
       [models[0].name, "iron man", 7.5, 50],
      
    ], inputs=[model_name, prompt, guidance, steps, seed], outputs=outputs, fn=inference, cache_examples=False)

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>Model by Linaqruf</p>
    </div>
    """)

print(f"Space built in {time.time() - start_time:.2f} seconds")

if not is_colab:
  demo.queue(concurrency_count=1)
demo.launch(debug=is_colab, share=is_colab)