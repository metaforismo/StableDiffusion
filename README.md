# My Learning Journey: Building Stable Diffusion from Scratch with PyTorch

## Overview
I recently completed a deep dive into Stable Diffusion by implementing it from scratch using PyTorch. This experience gave me a comprehensive understanding of how modern text-to-image AI systems work at a fundamental level.

## What I Learned About Stable Diffusion

### Core Architecture
Stable Diffusion is a **latent diffusion model** that generates images from text prompts. I learned that it consists of three main components working together:

1. **VAE (Variational Autoencoder)**: Compresses images into a latent space and reconstructs them
2. **CLIP Text Encoder**: Converts text prompts into embeddings
3. **U-Net**: The diffusion model that denoises latents iteratively

### 1. Text Encoding with CLIP

I implemented the CLIP (Contrastive Language-Image Pre-training) encoder to transform text prompts into meaningful embeddings:

- **Token & Position Embeddings**: The model uses a vocabulary of 49,408 tokens and supports sequences up to 77 tokens
- **Transformer Architecture**: 12 layers of self-attention with 12 heads each, operating on 768-dimensional embeddings
- **QuickGELU Activation**: I learned about this special activation function `x * sigmoid(1.702 * x)` used instead of standard GELU
- **Pre-LayerNorm**: The model uses pre-normalization (LayerNorm before attention) rather than post-normalization

The CLIP encoder outputs shape `(batch, 77, 768)` - a sequence of 77 token embeddings, each 768-dimensional.

### 2. VAE: Image Compression and Reconstruction

The VAE was fascinating to implement because it works in latent space rather than pixel space:

**Encoder**:
- Takes RGB images `(B, 3, 512, 512)`
- Compresses them 8x in each spatial dimension
- Outputs latents `(B, 4, 64, 64)` - reducing data by 48x!
- This compression makes diffusion computationally feasible

**Decoder**:
- Takes the denoised latents `(B, 4, 64, 64)`
- Reconstructs back to full resolution `(B, 3, 512, 512)`
- Uses transposed convolutions and upsampling

I learned that working in latent space is the key innovation that makes Stable Diffusion efficient compared to pixel-space diffusion models.

### 3. U-Net Diffusion Model

The U-Net is the heart of the system. I implemented it with these key insights:

**Architecture Pattern**:
- **Encoder path**: Progressively downsamples from 320 → 640 → 1280 channels while reducing spatial dimensions
- **Bottleneck**: Processes features at the lowest resolution (H/64, W/64)
- **Decoder path**: Progressively upsamples back with skip connections from encoder

**Two Types of Blocks**:

1. **Residual Blocks** (for temporal conditioning):
   - Use GroupNorm (groups of 32)
   - Integrate time embeddings: I learned how timestep information gets injected via linear projection and addition
   - Apply SiLU activation

2. **Attention Blocks** (for spatial and text conditioning):
   - **Self-Attention**: Relates different parts of the image to each other
   - **Cross-Attention**: This is where the magic happens - text embeddings guide image generation!
   - **GeGLU FFN**: Learned about Gated Linear Units for the feed-forward network

**Skip Connections**: The decoder concatenates encoder features, doubling channel counts (e.g., 1280 → 2560), which helps preserve spatial details.

### 4. Time Embeddings

I implemented sinusoidal time embeddings following the "Attention is All You Need" positional encoding:
```
freqs = 10000^(-i/160) for i in [0, 160)
embedding = [cos(t * freqs), sin(t * freqs)]
```
This creates a 320-dimensional representation of the timestep, which gets expanded to 1280 dimensions via MLPs.

### 5. DDPM Sampling

I learned the denoising diffusion probabilistic model (DDPM) sampling process:

**Forward Diffusion** (Training):
- Gradually adds Gaussian noise to images over T timesteps
- Follows a variance schedule

**Reverse Diffusion** (Inference):
- Starts from pure noise `(B, 4, 64, 64)`
- Iteratively predicts and removes noise for 50 steps
- Each step: `latent_t-1 = (latent_t - predicted_noise) / scaling_factor + noise`

**Strength Parameter**: For image-to-image, I learned that "strength" controls how much to modify the input - higher strength means starting from a noisier latent.

### 6. Classifier-Free Guidance (CFG)

One of the most important concepts I learned was CFG for controllable generation:

**How it works**:
1. Run the model twice in parallel:
   - Once with the text prompt (conditional)
   - Once without text/empty prompt (unconditional)
2. Interpolate: `output = cfg_scale * (cond - uncond) + uncond`
3. Higher cfg_scale (e.g., 7.5) makes the model follow the prompt more closely

This is why the U-Net processes batches of 2 when CFG is enabled - it's computing both versions simultaneously!

### 7. The Complete Pipeline

I implemented the full generation pipeline:

**Text-to-Image**:
1. Tokenize prompt → CLIP encoding `(1, 77, 768)`
2. Initialize random latent `(1, 4, 64, 64)`
3. For each timestep (50 iterations):
   - Get time embedding
   - U-Net predicts noise
   - Apply CFG if enabled
   - Denoise one step
4. VAE decode latents to image
5. Rescale from `[-1, 1]` to `[0, 255]`

**Image-to-Image**:
1. Encode input image to latent space
2. Add noise based on strength
3. Start denoising from this noisy latent (same as steps 3-5 above)

### 8. Technical Details I Mastered

**GroupNorm**: I learned why GroupNorm (32 groups) is preferred over BatchNorm for diffusion models - it's more stable with small batch sizes.

**Attention Mechanics**:
- Self-attention: `Attention(Q, K, V)` where Q=K=V from the same sequence
- Cross-attention: Q from image features, K and V from text embeddings
- Masking: CLIP uses causal masking for autoregressive text processing

**Shape Transformations**: I became comfortable with complex shape manipulations:
- `(B, C, H, W) → (B, H*W, C)` for attention
- Broadcasting time embeddings with `unsqueeze(-1).unsqueeze(-1)`
- Chunking batches for CFG

**Weight Loading**: I learned about mapping pretrained Stable Diffusion weights to my custom architecture - this taught me about model parameter structure and naming conventions.

## Key Insights

1. **Latent Diffusion is Efficient**: Operating in compressed latent space (64×64×4) instead of pixel space (512×512×3) makes generation ~48x more efficient

2. **Cross-Attention Connects Text and Image**: This is the mechanism that allows text prompts to guide image generation - Q from images attends to K,V from text

3. **Iterative Refinement**: 50 denoising steps gradually transform noise into a coherent image - each step makes small improvements

4. **Conditioning is Everything**: Time embeddings tell the model "how noisy is this?", text embeddings tell it "what to generate", and CFG tells it "how much to follow the prompt"

5. **Architecture Symmetry**: The U-Net's encoder-decoder symmetry with skip connections preserves information while allowing the model to process at multiple scales

## Practical Skills Gained

- Implementing complex PyTorch architectures from scratch
- Understanding attention mechanisms deeply (self, cross, causal)
- Working with pretrained model weights
- Designing efficient inference pipelines
- Managing GPU memory (offloading models to CPU when idle)
- Using noise schedules and sampling algorithms

## What's Next

Now that I understand Stable Diffusion's internals, I want to explore:
- DDIM and other faster samplers
- ControlNet for structural conditioning
- LoRA for efficient fine-tuning
- Stable Diffusion XL architecture improvements
- Latent consistency models

This project transformed my understanding of generative AI from a black box to a system I can build, modify, and reason about deeply.
