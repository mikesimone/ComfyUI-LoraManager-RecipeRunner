# LoraManager Recipe Runner (WAN 2.2)

A ComfyUI node that turns a [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) recipe into
everything a WAN 2.2 image-to-video run with the two experts (high-noise and low-noise) needs: prompt, negative,
sampler settings, seed, and the recipe's LoRAs already split between the two experts.

## Requirements

- ComfyUI
- ComfyUI-Lora-Manager (provides the recipes, the `.metadata.json` files used to find LoRAs by Civitai version, the
  LoRA stack nodes and Send to Workflow)

## Install

ComfyUI-Manager: search for "LoraManager Recipe Runner". Or clone this repo into `ComfyUI/custom_nodes/` and restart.

## Usage

1. Add **LoraManager Recipe Runner (WAN 2.2)** (category `LoraManager`).
2. Pick a recipe:
   - **auto**: wire a LoraManager LoRA stack into `received_loras`, then click **Send to Workflow** on a recipe in
     LoraManager. The node finds the recipe whose LoRAs match the sent ones best (by Civitai version id). If the
     workflow has only one LoraManager stack, Send to Workflow doesn't ask which node to fill.
   - **dropdown**: pick a recipe (newest first). Its LoRAs are resolved from your local library; nothing needs to be
     sent. Press R (refresh node definitions) to see recipes saved after ComfyUI started.
3. Wire the outputs:
   - `high_loras` into a LoRA stack loader on the high-noise model, `low_loras` into one on the low-noise model
   - `positive` / `negative` into your text encoders
   - `steps`, `cfg`, `sampler_name`, `scheduler`, `seed` into both KSamplerAdvanced nodes; `high_end_step` is the
     step where the high-noise sampler stops and the low-noise one starts (steps // 2)
   - `shift` into ModelSamplingSD3; `frames` / `fps` into your latent and video nodes
   - `summary` into a text display to see what will run

## High / low routing

WAN 2.2 LoRAs usually come as a pair of files. The node decides by file name (and the Civitai version name):

- a name containing `high` (or `HN`, `_H`, `-H`) goes on the high-noise expert only
- a name containing `low` (or `LN`, `_L`, `-L`) goes on the low-noise expert only
- anything else (single-file LoRAs, e.g. WAN 2.1 LoRAs) goes on both

When LoRAs are sent from LoraManager, the sent stack wins over the recipe's list, so strengths you changed in the
stack are kept.

## Settings resolution

Recipe values win. When a recipe lacks a value, the node's fallback widgets are used (defaults: 8 steps, CFG 1,
euler / simple, shift 5, 81 frames at 16 fps). `prompt_override` / `negative_override`, when filled in, win over the
recipe. `style_token` (empty by default) is prepended to the positive, e.g. for a checkpoint's style trigger.

The recipes folder is found the way LoraManager finds it: `recipes_path` from LoraManager's `settings.json`
(`LORA_MANAGER_SETTINGS_DIR`, the portable copy in its node folder, or the user config dir), otherwise
`<first LoRA root>/recipes`. LoRA files are resolved through ComfyUI's `loras` folder paths.

## Speed LoRAs

Many WAN 2.2 workflows use a step-distill ("lightx2v" / "Lightning") LoRA for 4-8 step generation at CFG 1. This
node doesn't add one; put it in your own loaders. If a recipe was made without one and records a CFG above 1, either
drop the speed LoRA or override the CFG. Checkpoints that have a speed LoRA merged in usually ask you not to add
another.

## Limits

- LoraManager's Send to Workflow carries only the LoRA list; prompt and generation settings come from the recipe file.
- Recipes don't record frame count or fps, and rarely shift, so those come from the widgets.
- Recipe LoRAs that aren't in your local library are listed in the summary and skipped.

## License

MIT
