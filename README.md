# ComfyUI-NAG-MiniMaxH3

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for the **MiniMax H3** audio-video model in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG makes negative prompts work in attention space instead of (or together with) CFG: it works with few-step and distilled models where CFG is off, and adds negative control on top of CFG in multi-step sampling.

The nodes mirror [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG), which does not support MiniMax H3. Both packs can be installed side by side; the node names do not clash.

Paper: https://arxiv.org/abs/2505.21179

> **Status:** early release. Please report problems in [Issues](https://github.com/BenderBlender/ComfyUI-NAG-MiniMaxH3/issues).

## Requirements

- A ComfyUI version with native MiniMax H3 support (`comfy/ldm/minimax`).
- No extra Python packages.

## Installation

### ComfyUI Manager

`Manager` → `Install via Git URL` → paste `https://github.com/BenderBlender/ComfyUI-NAG-MiniMaxH3`, then restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BenderBlender/ComfyUI-NAG-MiniMaxH3
```

Restart ComfyUI.

## Nodes

| Stock node | NAG node | Category |
| --- | --- | --- |
| `KSampler` | `KSamplerWithNAG (MiniMax H3)` | `sampling` |
| `KSampler (Advanced)` | `KSamplerWithNAG (Advanced, MiniMax H3)` | `sampling` |
| `SamplerCustom` | `SamplerCustomWithNAG (MiniMax H3)` | `sampling/custom_sampling` |
| `BasicGuider` | `NAGGuider (MiniMax H3)` | `sampling/custom_sampling/guiders` |
| `CFGGuider` | `NAGCFGGuider (MiniMax H3)` | `sampling/custom_sampling/guiders` |
| — | `Apply NAG (MiniMax H3)` | `model/patch/minimax` |

`Apply NAG (MiniMax H3)` patches the `MODEL` itself, so every sampler or guider after it samples with NAG. Applying NAG a second time replaces the earlier settings.

## Example workflows

Example workflows are available in the [`workflows`](workflows) directory. They are built from ComfyUI's own Pika H3 Flex (MiniMax H3) templates and run three branches side by side with the same models, inputs, seed, sampler and sigmas, so the videos differ only by guidance:

1. **Without NAG**: the original template (`BasicGuider`).
2. **NAG, empty negative prompt**: `NAGGuider (MiniMax H3)` with an empty NAG negative prompt.
3. **NAG, negative prompt**: `NAGGuider (MiniMax H3)` with a filled NAG negative prompt.

| Workflow | Task | ComfyUI template |
| --- | --- | --- |
| [NAG-MiniMax-H3-Pika-Flex-FL2VA](workflows/NAG-MiniMax-H3-Pika-Flex-FL2VA-ComfyUI-Workflow.json) | first and last frame to video | Pika H3 Flex: FL2VA |
| [NAG-MiniMax-H3-Pika-Flex-REF2VA](workflows/NAG-MiniMax-H3-Pika-Flex-REF2VA-ComfyUI-Workflow.json) | reference images to video | Pika H3 Flex: Reference to Video |

The model files and input images are linked in each workflow's `Note: Model Links`. The FL2VA template's subgraph is unpacked into a flat graph so the three branches are visible. Running all three branches takes three to four times as long as the template (NAG adds a second attention pass): mute the ones you don't need (Ctrl+M).

## Usage

1. Build your usual MiniMax H3 workflow (text-to-video, image-to-video, reference-to-video or keyframe guides).
2. Replace the sampler or guider with its NAG version from the table above.
3. Encode a negative prompt with the H3 text encoder (`CLIPTextEncode`) and connect it to `nag_negative`.
4. Start with the defaults, then tune as described below.

`nag_negative` is independent of the CFG `negative` input: with `cfg = 1` (CFG off) NAG alone handles the negative prompt, with `cfg > 1` both are applied.

## Key inputs

When working with a new model or task, first find a good combination of `nag_tau` and `nag_alpha` that makes the negative guidance effective without artifacts. Then keep them fixed and tune only `nag_scale`.

- `nag_scale`: the scale for attention feature extrapolation. Higher values result in stronger negative guidance. NAG is off at `nag_scale <= 1`.
- `nag_tau`: the normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
- `nag_sigma_end`: NAG is active only while sigma >= `nag_sigma_end`. Raise it to skip NAG on the last steps and save time.

The defaults are the ones of ComfyUI-NAG (`nag_scale` 5.0, `nag_tau` 2.5, `nag_alpha` 0.25, `nag_sigma_end` 0.0). The guiders have no `latent_image` input: H3 samples one video at a time.

### Rule of thumb

- For image-to-video, reference-to-video and keyframe guides, use lower `nag_tau` and `nag_alpha` to keep the references faithful.
- For multi-step sampling with higher CFG, also prefer lower `nag_tau` and `nag_alpha`.
- For few-step models, higher `nag_tau` and `nag_alpha` give stronger negative guidance.

## How it works

MiniMax H3 is a single-stream DiT: text, guide/reference rows, audio and video tokens are packed into one sequence with one attention per block. NAG runs a second, negative-prompt text stream next to it, the way ComfyUI-NAG handles Flux single-stream blocks:

- the negative text rows are appended to the packed sequence of every block;
- the video, audio and guide rows attend once with the positive text and once with the negative text, and their attention output is replaced with the NAG mix of the two;
- the negative text sits where the positive text ends on the RoPE time axis, which is the layout a sequence built around the negative prompt would have.

NAG is installed through ComfyUI's model patches (a diffusion-model wrapper plus `dit` block patches), so the stock samplers keep handling the nested video+audio latent.

## Compatibility and performance

- **Cost:** while NAG is active, every block runs a second attention over the non-text rows, about 2x the attention time of the positive pass. Use `nag_sigma_end` to stop it early.
- **Apply MiniMax H3 Fun ControlNet:** works in any order; NAG runs right around each block, inside other block patches.
- **Model Sparse Attention:** it replaces the block patches of every node before it, so put it **before** NAG. On the steps where NAG is active, the negative pass runs dense, and the positive pass goes through the sparse attention's generic kernel (dense with VSA).
- **Batch size:** 1, like MiniMax H3 itself.

## Credits

- [Normalized Attention Guidance](https://github.com/ChenDarYen/Normalized-Attention-Guidance) by Dar-Yen Chen et al.
- [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG) by Dar-Yen Chen (MIT): the node design and the NAG formula used here.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and its native MiniMax H3 implementation.
- The example workflows are based on the Pika H3 Flex templates from [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (Comfy-Org, MIT) and use [Pika H3 Flex](https://github.com/Pika-Labs/Pika-H3-Flex).

## License

[MIT](LICENSE)
