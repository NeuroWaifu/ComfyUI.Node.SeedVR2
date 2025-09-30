# SeedVR2 for ComfyUI

Official ComfyUI implementation of SeedVR2 - One-Step Video Restoration via Diffusion Adversarial Post-Training.

Based on code from:
- https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- https://github.com/ByteDance-Seed/SeedVR

## Key Differences from numz Version

This implementation includes several improvements over the original numz version:
- **Added support for `extra_model_paths.yaml`** - Allows flexible model path configuration
- **Separate VAE field** - VAE model selection is now explicit for better control
- **Removed automatic model downloader** - Provides users with full control over model management

## Overview

SeedVR2 is a state-of-the-art video upscaling model that achieves high-quality video restoration in a single step. This implementation provides seamless integration with ComfyUI, allowing you to enhance video quality with minimal computational overhead compared to traditional multi-step diffusion models.

### Available Nodes

#### SeedVR2 Video Upscaler
![SeedVR2 Video Upscaler Node](.github/SeedVR2%20Video%20Upscaler.png)

The main node for video/image upscaling. Processes input frames using the SeedVR2 diffusion model to produce high-quality upscaled output. Supports batch processing for temporal consistency and various model configurations.

#### SeedVR2 BlockSwap Config
![SeedVR2 BlockSwap Config Node](.github/SeedVR2%20BlockSwap%20Config.png)

Optional configuration node for memory optimization. Enables running large models on limited VRAM by dynamically swapping transformer blocks between GPU and CPU/RAM. Essential for consumer GPUs with less than 24GB VRAM.

### Key Features

- **One-Step Video Restoration**: Process videos with any resolution in a single step without relying on additional diffusion priors
- **Multiple Model Variants**: Support for 3B and 7B parameter models in both FP16 and FP8 formats
- **Advanced Memory Management**: BlockSwap technology for running large models on consumer GPUs
- **Temporal Consistency**: Maintains video coherence across frames with batch processing
- **ComfyUI Integration**: Easy-to-use nodes with visual workflow support

## Installation

### Prerequisites

- ComfyUI (latest version recommended)
- Python 3.10+ (tested with 3.12.9)
- NVIDIA GPU with:
  - Minimum 18GB VRAM for 3B models
  - Minimum 24GB VRAM for 7B models
  - BlockSwap can reduce requirements significantly

### Installation Steps

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/NeuroWaifu/ComfyUI.Node.SeedVR2.git
```

3. Restart ComfyUI - dependencies will be installed automatically from requirements.txt

   Alternatively, you can manually install dependencies:
```bash
cd ComfyUI.Node.SeedVR2
pip install -r requirements.txt
```

4. Download model weights:

Create a `SEEDVR2` folder in your ComfyUI models directory and download the following models:

- **3B Models**:
  - `seedvr2_ema_3b_fp16.safetensors` - Full precision (recommended)
  - `seedvr2_ema_3b_fp8_e4m3fn.safetensors` - Reduced precision for lower VRAM

- **7B Models**:
  - `seedvr2_ema_7b_fp16.safetensors` - Full precision (best quality)
  - `seedvr2_ema_7b_fp8_e4m3fn.safetensors` - Reduced precision (quality issues reported)

Models can be downloaded from: [Hugging Face](https://huggingface.co/numz/SeedVR2_comfyUI)

5. Download VAE model:

Place your VAE model (e.g., from Stable Diffusion) in the ComfyUI VAE models folder.

## Usage

### Basic Workflow

1. In ComfyUI, locate the **SeedVR2 Video Upscaler** node in the node menu under the SEEDVR2 category
2. Connect your input images/video frames to the node
3. Configure the following parameters:
   - **Model**: Select your downloaded SeedVR2 model
   - **VAE**: Select your VAE model
   - **Seed**: Random seed for reproducibility
   - **Target Short Side**: Desired resolution (default: 1072)
   - **Batch Size**: Number of frames to process together (minimum 5 for temporal consistency)
   - **Preserve VRAM**: Enable to reduce memory usage

### Advanced Usage with BlockSwap

For GPUs with limited VRAM, use the **SeedVR2 BlockSwap Config** node:

1. Add the BlockSwap Config node to your workflow
2. Connect it to the SeedVR2 Video Upscaler's `block_swap_config` input
3. Configure BlockSwap parameters:
   - **blocks_to_swap**: Number of transformer blocks to offload (0-36)
   - **use_non_blocking**: Enable asynchronous GPU transfers (recommended)
   - **offload_io_components**: Offload embeddings to CPU for extra savings
   - **cache_model**: Keep model in RAM between runs
   - **enable_debug**: Show detailed memory usage

### Recommended Settings

#### For 24GB VRAM GPUs:
- Model: 3B FP16
- Batch Size: 9-13
- BlockSwap: 0 (disabled)

#### For 16GB VRAM GPUs:
- Model: 3B FP8
- Batch Size: 5
- BlockSwap: 16-20 blocks

#### For 12GB VRAM GPUs:
- Model: 3B FP8
- Batch Size: 5
- BlockSwap: 24-32 blocks
- Enable offload_io_components

## Parameters

### Video Upscaler Node

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| images | IMAGE | Required | - | Input video frames |
| model | COMBO | Required | - | SeedVR2 model selection |
| vae | COMBO | Required | - | VAE model selection |
| seed | INT | 100 | 0-2^32 | Random seed for generation |
| target_short_side | INT | 1072 | 16-4320 | Target resolution (short side) |
| batch_size | INT | 5 | 1-2048 | Frames per batch (use 4n+1 format) |
| preserve_vram | BOOLEAN | False | - | Enable memory optimization |
| block_swap_config | Optional | None | - | BlockSwap configuration |

### BlockSwap Config Node

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| blocks_to_swap | INT | 16 | Number of blocks to offload (0=disabled) |
| use_non_blocking | BOOLEAN | True | Use async GPU transfers |
| offload_io_components | BOOLEAN | False | Offload embeddings/IO layers |
| cache_model | BOOLEAN | False | Keep model in RAM between runs |
| enable_debug | BOOLEAN | False | Show memory/timing stats |

## Performance Tips

1. **Batch Size**: 
   - Minimum 5 frames required for temporal consistency
   - Use 4n+1 format (5, 9, 13, 17...) for optimal performance
   - Higher batch sizes improve quality but require more VRAM

2. **BlockSwap Tuning**:
   - Start with blocks_to_swap=16 and increase if OOM occurs
   - Each additional block reduces VRAM by ~200-400MB
   - Performance impact is minimal up to 20 blocks

3. **Model Selection**:
   - 3B FP16: Best balance of quality and performance
   - 7B FP16: Highest quality but requires significant VRAM
   - FP8 models: Lower VRAM usage but slight quality reduction

## Troubleshooting

### Out of Memory (OOM) Errors

1. Enable BlockSwap and increase blocks_to_swap
2. Reduce batch_size to minimum (5)
3. Switch to FP8 model variant
4. Enable offload_io_components in BlockSwap
5. Close other GPU applications

### Poor Output Quality

1. Ensure batch_size is at least 5 for temporal consistency
2. Use FP16 models instead of FP8
3. Try different seed values
4. Check input video quality (very low quality inputs may not upscale well)

### Slow Performance

1. Reduce blocks_to_swap if you have spare VRAM
2. Enable use_non_blocking in BlockSwap
3. Use cache_model for batch processing
4. Ensure CUDA and PyTorch are properly configured

### Memory Optimization

The BlockSwap system dynamically manages model components:
- Transformer blocks can be offloaded to CPU/RAM
- Asynchronous transfers minimize performance impact
- Smart caching reduces redundant operations

## Acknowledgments

- Original SeedVR2 implementation by ByteDance-Seed
- BlockSwap integration by Adrien Toupet from AInVFX
- ComfyUI framework by comfyanonymous

## Links

- [Official Paper](https://arxiv.org/abs/2506.05301)
- [Project Page](https://iceclear.github.io/projects/seedvr2/)
- [Model Weights](https://huggingface.co/numz/SeedVR2_comfyUI)
