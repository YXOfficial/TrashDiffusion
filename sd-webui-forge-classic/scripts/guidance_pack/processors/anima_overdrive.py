import logging
import types
import gradio as gr
import torch

from ..base import GuidanceProcessor
from ..registry import register_processor
from core.forge_compat import get_unet, set_unet


_STATE = {"gain": 1.0, "temp": 1.0, "blocks": set(), "qwen_fix": False, "qwen_scale": 1.0}
_QWEN_ORIG = {}


def _parse_blocks(blocks_str, total_blocks):
    blocks = []
    for part in str(blocks_str).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                s, e = part.split("-", 1)
                blocks.extend(range(int(s), min(int(e) + 1, total_blocks)))
            except ValueError:
                pass
        else:
            try:
                idx = int(part)
                if 0 <= idx < total_blocks:
                    blocks.append(idx)
            except ValueError:
                pass
    if not blocks:
        blocks = list(range(total_blocks))
    return set(blocks)


def _ensure_cross_patched(model):
    # Latest backend/nn/anima.py: Block.cross_attn is SelfCrossAttention
    # with is_SelfAttn == False. Two levers, both single-pass:
    #  - temp on K: scales logits BEFORE softmax, so it survives the next
    #    block's LayerNorm (this is the pre-wash intervention; output gain
    #    alone gets washed out).
    #  - gain on output: mild residual push, kept small.
    for name, module in model.named_modules():
        if module.__class__.__name__ != "SelfCrossAttention":
            continue
        if getattr(module, "is_SelfAttn", None) is not False:
            continue
        if getattr(module, "_overdrive_patched", False):
            continue
        parts = name.split(".")
        try:
            # names look like "blocks.3.cross_attn"
            bi = parts.index("blocks")
            block_idx = int(parts[bi + 1])
        except (ValueError, IndexError):
            continue

        orig = module.compute_attention

        def make_patched(idx, original):
            def patched(this, q, k, v, transformer_options=None):
                if transformer_options is None:
                    transformer_options = {}
                t = _STATE["temp"]
                if t != 1.0 and idx in _STATE["blocks"]:
                    k = k * t
                out = original(q, k, v, transformer_options=transformer_options)
                g = _STATE["gain"]
                if g != 1.0 and idx in _STATE["blocks"]:
                    out = out * g
                return out

            return patched

        module.compute_attention = types.MethodType(make_patched(block_idx, orig), module)
        module._overdrive_patched = True


def _patch_legacy_anima(OldCls):
    # Old API: AnimaTextProcessingEngine with tokenize_line / process_tokens
    # / PromptChunk. Stock tokenize_line hardcoded qwen_multipliers=1.0.
    _QWEN_ORIG["tokenize_line"] = OldCls.tokenize_line
    _QWEN_ORIG["process_tokens"] = OldCls.process_tokens
    orig_process_tokens = OldCls.process_tokens

    def tokenize_line(self, line):
        from backend.text_processing import parsing
        from backend.text_processing.anima_engine import PromptChunk
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        qwen_tokenized, t5_tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk
            if not chunk.qwen_tokens:
                chunk.qwen_tokens.append(self.id_pad)
                chunk.qwen_multipliers.append(1.0)
            chunk.t5_tokens.append(self.id_end)
            chunk.t5_multipliers.append(1.0)
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (_, weight) in zip(qwen_tokenized, parsed):
            for token in tokens:
                chunk.qwen_tokens.append(token)
                chunk.qwen_multipliers.append(float(weight))

        for tokens, (text, weight) in zip(t5_tokenized, parsed):
            for token in tokens:
                chunk.t5_tokens.append(token)
                chunk.t5_multipliers.append(weight)

        if not chunks:
            next_chunk()

        return chunks

    def process_tokens(self, batch_tokens, batch_multipliers):
        z = orig_process_tokens(self, batch_tokens, batch_multipliers)
        if not _STATE["qwen_fix"]:
            return z
        try:
            s = float(_STATE["qwen_scale"])
            w = torch.as_tensor(batch_multipliers, dtype=torch.float32)
            if w.dim() != 2 or w.shape[0] != z.shape[0] or w.shape[1] != z.shape[1]:
                return z
            if s != 1.0:
                w = 1.0 + (w - 1.0) * s
            return (z.float() * w.to(z.device).unsqueeze(-1)).to(z.dtype)
        except Exception:
            return z

    OldCls.tokenize_line = tokenize_line
    OldCls.process_tokens = process_tokens


def _patch_qwen06(Qwen06Engine):
    # Latest API (ComfyUI v0.36 / forge-classic): Qwen06Engine.__call__
    # Stock code wipes Qwen weights:
    #   qwen_chunk = [[(x[0], 1.0) for x in inner] for inner in qwen_chunk]
    # so (word:1.3) never reaches Qwen. Preserve parsed weights instead.
    # encode_token_weights already implements (z - z_empty) * w + z_empty.
    if "call" in _QWEN_ORIG:
        return
    _QWEN_ORIG["call"] = Qwen06Engine.__call__
    orig_call = Qwen06Engine.__call__

    def patched_call(self, texts: list[str]):
        if not _STATE.get("qwen_fix", False):
            return orig_call(self, texts)
        try:
            from backend.args import dynamic_args
            from backend.text_processing import emphasis

            if any(emphasis.uses_emphasis(text) for text in texts) and self.emphasis.name in ("None", "Ignore"):
                dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

            n: bool = self.emphasis.name == "None"
            i: bool = self.emphasis.name == "Ignore"
            s: float = float(_STATE.get("qwen_scale", 1.0))

            zs: list[torch.Tensor] = []
            cache: dict[str, torch.Tensor] = {}

            for line in texts:
                if line in cache:
                    z = cache[line]
                else:
                    qwen_chunk = self.qwen_tokenizer.tokenize_with_weights(line, disable_weights=n)
                    t5_chunk = self.t5_tokenizer.tokenize_with_weights(line, disable_weights=n)

                    if not i:
                        if s != 1.0:
                            # Scale emphasis distance from 1.0: w' = 1 + (w-1)*s
                            scaled = []
                            for inner in qwen_chunk:
                                row = []
                                for x in inner:
                                    tok = x[0]
                                    try:
                                        w = float(x[1])
                                    except Exception:
                                        w = 1.0
                                    row.append((tok, 1.0 + (w - 1.0) * s))
                                scaled.append(row)
                            qwen_chunk = scaled
                        # else: keep parsed weights as-is (this is the fix)
                    else:
                        # Emphasis=Ignore: force 1.0 like stock T5 path does
                        qwen_chunk = [[(x[0], 1.0) for x in inner] for inner in qwen_chunk]

                    cond = self.text_encoder.encode_token_weights(qwen_chunk)[0]
                    ids = torch.tensor(list(map(lambda x: x[0], t5_chunk[0])), dtype=torch.int).unsqueeze(0)
                    weights = torch.tensor(list(map(lambda x: (1.0 if i else x[1]), t5_chunk[0]))).unsqueeze(0).unsqueeze(-1)

                    z = self._preprocess(cond, ids, weights)
                    cache[line] = z

                zs.append(z)

            return zs
        except Exception as e:
            logging.warning(f"Anima Overdrive: Qwen fix failed, fallback to stock: {e}")
            return orig_call(self, texts)

    Qwen06Engine.__call__ = patched_call


def _ensure_qwen_patched():
    # Support both old (AnimaTextProcessingEngine) and latest (Qwen06Engine).
    # Never raise: Qwen fix is optional, cross-attn boost must still work.
    if _QWEN_ORIG:
        return
    try:
        import backend.text_processing.anima_engine as anima_mod
    except Exception as e:
        logging.warning(f"Anima Overdrive: cannot import anima_engine, Qwen fix skipped: {e}")
        _QWEN_ORIG["skipped"] = True
        return

    OldCls = getattr(anima_mod, "AnimaTextProcessingEngine", None)
    if OldCls is not None and hasattr(OldCls, "tokenize_line") and hasattr(OldCls, "process_tokens"):
        try:
            _patch_legacy_anima(OldCls)
            logging.info("Anima Overdrive: patched legacy AnimaTextProcessingEngine")
        except Exception as e:
            logging.warning(f"Anima Overdrive: legacy Qwen patch failed: {e}")
            _QWEN_ORIG["skipped"] = True
        return

    NewCls = getattr(anima_mod, "Qwen06Engine", None)
    if NewCls is not None:
        try:
            _patch_qwen06(NewCls)
            logging.info("Anima Overdrive: patched Qwen06Engine (latest)")
        except Exception as e:
            logging.warning(f"Anima Overdrive: Qwen06 patch failed: {e}")
            _QWEN_ORIG["skipped"] = True
        return

    # Fallback: unknown future rename — try any class with qwen_tokenizer + encode path
    for attr_name in dir(anima_mod):
        try:
            cand = getattr(anima_mod, attr_name)
            if isinstance(cand, type) and hasattr(cand, "__call__") and "Qwen" in attr_name:
                _patch_qwen06(cand)
                logging.info(f"Anima Overdrive: patched {attr_name} (fallback)")
                return
        except Exception:
            continue
    logging.warning("Anima Overdrive: no known Anima text engine found, Qwen fix skipped")
    _QWEN_ORIG["skipped"] = True


class AnimaOverdriveProcessor(GuidanceProcessor):
    def name(self) -> str:
        return "Anima Overdrive (CFG-free prompt push)"

    def create_ui(self):
        with gr.Tab(label="Anima Overdrive"):
            gr.Markdown("### Anima Overdrive\nBoosts cross-attn in a single pass. Unlike CFG: no uncond, no second forward.")
            enabled = gr.Checkbox(label="Enable Anima Overdrive", value=False)
            boost = gr.Slider(label="Cross Boost (post, mild)", minimum=1.0, maximum=10.0, step=0.05, value=1.1)
            temp = gr.Slider(label="Logit Temp (pre-softmax, survives LayerNorm)", minimum=1.0, maximum=10.0, step=0.01, value=1.15)
            blocks_list = gr.Textbox(label="Anima Blocks (e.g. 0-9)", value="0-9")
            qwen_fix = gr.Checkbox(label="Enable Qwen Emphasis Fix (word:1.3 -> Qwen)", value=True)
            qwen_scale = gr.Slider(label="Emphasis Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
        return [enabled, boost, temp, blocks_list, qwen_fix, qwen_scale]

    def infotext_fields(self):
        return [
            "Anima Overdrive Enabled",
            "Anima Overdrive boost",
            "Anima Overdrive temp",
            "Anima Overdrive blocks",
            "Anima Overdrive Qwen fix",
            "Anima Overdrive Qwen scale",
        ]

    def process(self, p, enabled, boost, temp, blocks_list, qwen_fix, qwen_scale):
        _ensure_qwen_patched()
        qwen_fix_eff = bool(enabled and qwen_fix)
        _STATE["qwen_fix"] = qwen_fix_eff
        _STATE["qwen_scale"] = float(qwen_scale)
        if not enabled:
            _STATE["gain"] = 1.0
            _STATE["temp"] = 1.0
            return

        self.record_params(p, {
            "Anima Overdrive Enabled": True,
            "Anima Overdrive boost": float(boost),
            "Anima Overdrive temp": float(temp),
            "Anima Overdrive blocks": str(blocks_list),
            "Anima Overdrive Qwen fix": qwen_fix_eff,
            "Anima Overdrive Qwen scale": float(qwen_scale),
        })

        unet_patcher = get_unet(p)
        model = unet_patcher.model.diffusion_model

        if model.__class__.__name__ != "Anima":
            logging.warning("Anima Overdrive: not Anima, skip.")
            _STATE["gain"] = 1.0
            _STATE["temp"] = 1.0
            return

        _ensure_cross_patched(model)
        total_blocks = len(model.blocks)
        _STATE["blocks"] = _parse_blocks(blocks_list, total_blocks)
        _STATE["gain"] = float(boost)
        _STATE["temp"] = float(temp)
        logging.info(f"Anima Overdrive: gain={float(boost):.2f} temp={float(temp):.2f} blocks={sorted(_STATE['blocks'])} qwen_fix={bool(qwen_fix)}")

    def register_xyz(self, xyz_grid, set_guidance_value_func):
        pass


# NOTE: do NOT print here. Processor load status is reported centrally by
# guidance_pack_script._load_processors() so the Forge log shows one uniform
# "[TrashDiffusion]" summary (totals + reasons) instead of one print style
# per module.
register_processor(AnimaOverdriveProcessor)
