"""LoraManager Recipe Runner (WAN 2.2): turns a ComfyUI-Lora-Manager recipe into everything a WAN 2.2 two-expert
(high-noise / low-noise) run needs.

Two ways to pick the recipe:
  * "auto": click Send to Workflow on a recipe in LoraManager. Its LoRAs land in the LoraManager stack wired to
    `received_loras`; the node finds the recipe whose LoRAs match best (by Civitai version id).
  * pick a recipe from the dropdown (newest first); its LoRAs are then resolved from the local library, no Send
    needed. Refresh node definitions (R) to see recipes saved after ComfyUI started.

The recipes folder is found the way LoraManager finds it: `recipes_path` from LoraManager's settings.json, else
`<first LoRA root>/recipes`; LoRA files are resolved through ComfyUI's `loras` folder paths.
"""
import glob
import json
import os
import re
import sys

import comfy.samplers
import folder_paths

AUTO = "auto: match the LoRAs sent from LoraManager"
APP_NAME = "ComfyUI-LoRA-Manager"
SAMPLER_ALIASES = {"euler": "euler", "euler a": "euler_ancestral", "euler_a": "euler_ancestral", "uni_pc": "uni_pc",
                   "unipc": "uni_pc", "dpm++ 2m": "dpmpp_2m", "dpm++ 2m sde": "dpmpp_2m_sde", "dpm++ sde": "dpmpp_sde",
                   "lcm": "lcm", "ddim": "ddim", "heun": "heun", "dpm++ 2m karras": "dpmpp_2m"}
_version_index = {}


# ---------------------------------------------------------------- locations
def _lora_roots():
    try:
        return [p for p in folder_paths.get_folder_paths("loras") if os.path.isdir(p)]
    except Exception:
        return []


def _settings_candidates():
    """LoraManager's settings.json, in the order LoraManager itself uses: explicit override, portable copy in its
    node folder, then the platform user config dir."""
    out = []
    override = os.environ.get("LORA_MANAGER_SETTINGS_DIR")
    if override:
        out.append(os.path.join(os.path.abspath(os.path.expanduser(override)), "settings.json"))
    try:
        node_dirs = folder_paths.get_folder_paths("custom_nodes")
    except Exception:
        node_dirs = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for nd in node_dirs:
        for d in glob.glob(os.path.join(nd, "*")):
            if re.search(r"lora.?manager", os.path.basename(d), re.I):
                out.append(os.path.join(d, "settings.json"))
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    out.append(os.path.join(base, APP_NAME, "settings.json"))
    return out


def _load_settings():
    for p in _settings_candidates():
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            continue
    return {}


def recipes_dir():
    s = _load_settings()
    custom = s.get("recipes_path")
    if isinstance(custom, str) and custom.strip():
        return os.path.abspath(os.path.expanduser(custom.strip()))
    roots = [r for r in ((s.get("folder_paths") or {}).get("loras") or []) if isinstance(r, str) and r.strip()]
    if roots:
        return os.path.abspath(os.path.join(sorted(roots, key=str.casefold)[0], "recipes"))
    for root in sorted(_lora_roots(), key=str.casefold):
        if os.path.isdir(os.path.join(root, "recipes")):
            return os.path.join(root, "recipes")
    return ""


def _recipe_files():
    d = recipes_dir()
    if not d:
        return []
    return sorted(glob.glob(os.path.join(d, "*.recipe.json")), key=os.path.getmtime, reverse=True)


def _label(path, d):
    title = re.sub(r"\s+", " ", (d.get("title") or "").strip())[:60]
    return f"{os.path.basename(path)[:8]} | {title}"


def _recipe_options():
    out = []
    for f in _recipe_files():
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(_label(f, json.load(fh)))
        except (OSError, ValueError):
            continue
    return out


# ---------------------------------------------------------------- LoRAs
def _side(text):
    """Which WAN 2.2 expert a LoRA file belongs to, from its name: high, low, or single (use on both)."""
    t = (text or "").lower()
    if re.search(r"high|\bhn\b|_h\b|-h\b", t):
        return "high"
    if re.search(r"low|\bln\b|_l\b|-l\b", t):
        return "low"
    return "single"


def _local_path(p):
    p = str(p)
    if os.path.isabs(p):
        return p
    try:
        return folder_paths.get_full_path("loras", p) or p
    except Exception:
        return p


def _civitai_version(lora_path):
    meta = os.path.splitext(_local_path(lora_path))[0] + ".metadata.json"
    try:
        with open(meta, encoding="utf-8") as fh:
            c = json.load(fh).get("civitai") or {}
        return str(c["id"]) if c.get("id") else None
    except (OSError, ValueError, KeyError):
        return None


def _index():
    """Civitai version id -> local LoRA file, from LoraManager's .metadata.json files (cached by root mtimes)."""
    roots = _lora_roots()
    key = tuple((r, os.path.getmtime(r)) for r in roots)
    if _version_index.get("key") != key:
        idx = {}
        for root in roots:
            for meta in glob.glob(os.path.join(root, "**", "*.metadata.json"), recursive=True):
                try:
                    with open(meta, encoding="utf-8") as fh:
                        c = json.load(fh).get("civitai") or {}
                except (OSError, ValueError):
                    continue
                if c.get("id"):
                    sf = meta[:-len(".metadata.json")] + ".safetensors"
                    if os.path.exists(sf):
                        idx.setdefault(str(c["id"]), sf)
        _version_index.update(key=key, idx=idx)
    return _version_index["idx"]


def _num(v, cast):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def resolve(recipe, received_loras=None):
    """Pick the recipe (dropdown label or AUTO) and resolve its LoRAs. Returns (path, data, how, high, low, missing)."""
    received = list(received_loras or [])
    received_versions = {v for v in (_civitai_version(p) for p, *_ in received) if v}
    chosen, data, how = None, {}, ""
    files = _recipe_files()
    if recipe != AUTO:
        rid = recipe.split(" | ", 1)[0]
        chosen = next((f for f in files if os.path.basename(f).startswith(rid)), None)
        how = "picked from the dropdown" if chosen else "picked recipe not found"
    elif received_versions:
        best = (0.0, None)
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            vs = {str(l.get("modelVersionId")) for l in d.get("loras") or [] if l.get("modelVersionId") and not l.get("exclude")}
            if vs:
                score = len(vs & received_versions) / len(vs | received_versions)
                if score > best[0]:
                    best = (score, f)
        chosen = best[1]
        how = f"matched the sent LoRAs ({best[0]:.0%} overlap)" if chosen else "no recipe matched the sent LoRAs"
    else:
        how = "nothing sent and no recipe picked: fallback settings only"
    if chosen:
        with open(chosen, encoding="utf-8") as fh:
            data = json.load(fh)

    # LoRAs: the received stack wins (strengths may have been tweaked); otherwise resolve the recipe's own list
    loras, missing = [], []
    if received:
        for p, ms, cs in received:
            loras.append((p, ms, cs, _side(os.path.basename(str(p)))))
    elif data:
        idx = _index()
        for l in data.get("loras") or []:
            if l.get("exclude"):
                continue
            path = idx.get(str(l.get("modelVersionId")))
            if path:
                st = float(l.get("strength") or 1.0)
                loras.append((path, st, st, _side(f"{os.path.basename(path)} {l.get('modelVersionName')}")))
            else:
                missing.append(f"{l.get('modelName')} / {l.get('modelVersionName')}")
    high = [(p, ms, cs) for p, ms, cs, s in loras if s in ("high", "single")]
    low = [(p, ms, cs) for p, ms, cs, s in loras if s in ("low", "single")]
    return chosen, data, how, high, low, missing


class LoraManagerRecipeRunner:
    CATEGORY = "LoraManager"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS,
                    "INT", "INT", "FLOAT", "INT", "FLOAT", "LORA_STACK", "LORA_STACK", "STRING")
    RETURN_NAMES = ("positive", "negative", "steps", "cfg", "sampler_name", "scheduler", "seed", "high_end_step",
                    "shift", "frames", "fps", "high_loras", "low_loras", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": ([AUTO] + _recipe_options(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "Used when the recipe has no seed."}),
                "prompt_override": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Leave empty to use the recipe's prompt."}),
                "negative_override": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "Leave empty to use the recipe's negative."}),
                "style_token": ("STRING", {"default": "",
                                           "tooltip": "Optional text prepended to the positive (e.g. a checkpoint's style "
                                                      "trigger). Empty adds nothing."}),
                "fallback_steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "Used when the recipe has no steps."}),
                "fallback_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                           "tooltip": "Used when the recipe has no CFG."}),
                "fallback_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "fallback_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "fallback_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                             "tooltip": "Used when the recipe has no shift (recipes rarely record it)."}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 1000, "tooltip": "Recipes never record frame count."}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "tooltip": "Recipes never record fps."}),
            },
            "optional": {
                "received_loras": ("LORA_STACK", {"tooltip": "The LoraManager stack that Send to Workflow fills."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        files = _recipe_files()
        return f"{len(files)}:{max((os.path.getmtime(f) for f in files), default=0)}:{json.dumps(kwargs, default=str)[:2000]}"

    def run(self, recipe, seed, prompt_override, negative_override, style_token="", fallback_steps=8, fallback_cfg=1.0,
            fallback_sampler="euler", fallback_scheduler="simple", fallback_shift=5.0, frames=81, fps=16.0,
            received_loras=None):
        chosen, data, how, high, low, missing = resolve(recipe, received_loras)
        gp = data.get("gen_params") or {}

        steps = _num(gp.get("steps"), int) or fallback_steps
        cfg = _num(gp.get("cfg_scale"), float)
        cfg = fallback_cfg if cfg is None else cfg
        sampler = SAMPLER_ALIASES.get(str(gp.get("sampler") or "").strip().lower(), fallback_sampler)
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            sampler = fallback_sampler
        scheduler = str(gp.get("scheduler") or fallback_scheduler).lower()
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            scheduler = fallback_scheduler
        run_seed = _num(gp.get("seed"), int)
        run_seed = seed if run_seed is None else run_seed
        shift = _num(gp.get("shift"), float) or fallback_shift
        positive = prompt_override.strip() or (gp.get("prompt") or "").strip()
        negative = negative_override.strip() or (gp.get("negative_prompt") or "").strip()
        token = (style_token or "").strip()
        if token and not positive.lower().startswith(token.lower()):
            positive = f"{token}, {positive}" if positive else token

        def src(key):
            return "recipe" if gp.get(key) not in (None, "") else "fallback"

        lines = [f"Recipe: {_label(chosen, data) if chosen else '(none)'} ({how})",
                 f"Recipes folder: {recipes_dir() or '(not found)'}",
                 f"Source: {data.get('source_path') or '-'}", "",
                 f"HIGH-noise LoRAs ({len(high)}):"] + [f"  {os.path.basename(str(p))}  {ms}" for p, ms, _ in high] + \
                [f"LOW-noise LoRAs ({len(low)}):"] + [f"  {os.path.basename(str(p))}  {ms}" for p, ms, _ in low]
        if missing:
            lines += ["Not in the local library (skipped): " + "; ".join(missing)]
        lines += ["", f"Style token: {token or '(none)'}",
                  f"Steps {steps} ({src('steps')}), high noise 0-{steps // 2}, low noise {steps // 2}-end",
                  f"CFG {cfg} ({src('cfg_scale')}), sampler {sampler} ({src('sampler')}), scheduler {scheduler} ({src('scheduler')})",
                  f"Seed {run_seed} ({'recipe' if gp.get('seed') not in (None, '') else 'node'}), shift {shift} ({src('shift')}), "
                  f"{frames} frames at {fps:g} fps",
                  "", "Prompt: " + ("override" if prompt_override.strip() else "recipe" if gp.get("prompt") else "EMPTY"),
                  "Negative: " + ("override" if negative_override.strip() else "recipe" if gp.get("negative_prompt") else "empty")]
        return (positive, negative, steps, cfg, sampler, scheduler, run_seed, steps // 2, shift, frames, fps, high, low,
                "\n".join(lines))


NODE_CLASS_MAPPINGS = {"LoraManagerRecipeRunner": LoraManagerRecipeRunner}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraManagerRecipeRunner": "LoraManager Recipe Runner (WAN 2.2)"}
