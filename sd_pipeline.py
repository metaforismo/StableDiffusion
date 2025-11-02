import torch
import numpy as np
from tqdm import tqdm
from ddpm_sampler import DDPMSampler

HEIGHT=512
WIDTH = 512
LATENT_HEIGHT = HEIGHT//8
LATENT_WIDTH = WIDTH//8

def rescale(x, old_range, new_range, clamp=False):
    # very basic rescaling function
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max-new_min)/(old_max-old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # using the same attention is all you need paper
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def generate(prompt: str, neg_prompt: str, input_img=None,
            strength=0.8, cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_infer_steps=50, models={}, seed=None,
            device=None, idle_device=None, tokenizer=None):
    with torch.no_grad(): # not training

        if not(0<strength<=1):
            raise ValueError("keep it b/w 0 and 1")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip = clip.to(device)

        # if classifier-free guidance:
        if cfg:
            # convert prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # tokens to tensor (B, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # input to clip: (B, seq_len) --> (B, seq_len, n_embd)
            cond_context = clip(cond_tokens)

            neg_tokens = tokenizer.batch_encode_plus([neg_prompt], padding="max_length", max_length=77).input_ids
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
            # (B, seq_len) --> (B, seq_len, n_embd)
            neg_context = clip(neg_tokens)
            # concatenate the prompt + negative prompt
            # (2, seq_len, n_embd) = (2, 77, 768)
            context = torch.cat([cond_context, neg_context])
        # if no cfg
        else:
            # convert to tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (B, seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, Seq_Len) -> (B, seq_Len, n_embd) = (1, 77, 768)
            context = clip(tokens)
        to_idle(clip) # offload after using to cpu

        # sampler
        if sampler_name=='ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_infer_steps)
        else:
            raise ValueError(f"unknwon sampler {sampler_name}")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        # for image-to-image tasks
        if input_img:
            enc = models["encoder"]
            enc.to(device)

            input_img_tensor = input_img.resize((WIDTH, HEIGHT))
            # (H, W, C)
            input_img_tensor = np.array(input_img_tensor)
            input_img_tensor = torch.tensor(input_img_tensor, dtype=torch.float32, device=device)
            # we want the pixel values between [-1,1] instead of [0,255] to input to the UNet
            input_img_tensor = rescale(input_img_tensor, (0,255), (-1,1))
            # (H, W, C) --> (B, H, W, C)
            input_img_tensor = input_img_tensor.unsqueeze(0)
            # (B, H, W, C) --> (B, C, H, W)
            input_img_tensor = input_img_tensor.permute(0, 3, 1, 2)

            enc_noise = torch.randn(latents_shape, generator=generator, device=device)

            # input to the encoder of VAE
            latents = enc(input_img_tensor, enc_noise)
            # adding noise after getting latent according to strength
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(enc) # offload to gpu
        # if text-to-image
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # to (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            # (B, 4, Latents_Height, Latents_Width)
            model_input = latents

            if cfg:
                # (B, 4, Latent_Height, Latent_Width) --> (2*B, 4, Latent_Height, Latent_Width)
                # make two copies for with and without prompt
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise by unet
            model_output = diffusion(model_input, context, time_embedding)
            # since we'll get a batch of 2 if cfg
            if cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond-output_uncond) + output_uncond # weighing the two according to the scaler of cfg

            # remove the predicted noise
            latents = sampler.step(timestep, latents, model_output)
        to_idle(diffusion)

        # we have the noise-removed/transformed latents, now it's time to decode
        dec = models["decoder"]
        dec.to(device)

        img = dec(latents)
        to_idle(dec)

        img = rescale(img, (-1,1), (0,255), clamp=True) # rescale back to rgb
        # (B, C, H, W) --> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)
        img = img.to('cpu', torch.uint8).numpy()
        return img[0]
