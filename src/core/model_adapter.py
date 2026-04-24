from __future__ import annotations

import importlib
import math
from typing import Dict, List

import torch
import tiktoken
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


QWEN_PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN_ENDOFTEXT = "<|endoftext|>"
QWEN_IMSTART = "<|im_start|>"
QWEN_IMEND = "<|im_end|>"
QWEN_EXTRAS = tuple(f"<|extra_{i}|>" for i in range(205))
QWEN_SPECIAL_START_ID = 151643
QWEN_SPECIAL_TOKENS = tuple(
    enumerate(
        (
            QWEN_ENDOFTEXT,
            QWEN_IMSTART,
            QWEN_IMEND,
        ) + QWEN_EXTRAS,
        start=QWEN_SPECIAL_START_ID,
    )
)


class LocalQwenTokenizer:
    def __init__(self, vocab_file: str, errors: str = "replace"):
        self.errors = errors
        self.mergeable_ranks = self._load_tiktoken_bpe(vocab_file)
        self.special_tokens = {token: index for index, token in QWEN_SPECIAL_TOKENS}
        self.tokenizer = tiktoken.Encoding(
            "Qwen",
            pat_str=QWEN_PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.eod_id = self.tokenizer.eot_token

    @staticmethod
    def _load_tiktoken_bpe(vocab_file: str):
        import base64

        with open(vocab_file, "rb") as f:
            contents = f.read()
        return {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=self.errors)


class CausalModelAdapter:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.hf_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model_type = self.hf_config.model_type
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()
        self.model_config = self.model.config
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_heads = int(self.model.config.num_attention_heads)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self.vocab_size = int(self.model.config.vocab_size)
        self.head_dim = self.hidden_size // self.num_heads

        if self.model_type == "opt":
            self.backbone = self.model.model.decoder
            self.layers = self.backbone.layers
        elif self.model_type == "qwen":
            self.backbone = self.model.transformer
            self.layers = self.backbone.h
            self.head_dim = int(self.model.config.kv_channels)
            self.qwen_module = importlib.import_module(type(self.model).__module__)
        else:
            raise ValueError(f"Unsupported model type for now: {self.model_type}")

    def _load_tokenizer(self):
        if self.model_type == "qwen":
            return LocalQwenTokenizer(f"{self.model_path}/qwen.tiktoken")
        return AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

    def _load_model(self):
        common_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.model_type == "qwen":
            qwen_kwargs = {
                "use_flash_attn": False,
            }
            if self.device == "cpu":
                qwen_kwargs["fp32"] = True
            elif self.dtype == torch.float16:
                qwen_kwargs["fp16"] = True
            else:
                qwen_kwargs["fp32"] = True
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **common_kwargs,
                **qwen_kwargs,
            )
            self._patch_qwen_runtime_compatibility(model)
            if self.device != "cpu":
                model = model.to(self.device)
            return model

        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            **common_kwargs,
        ).to(self.device)

    def get_model_spec(self) -> Dict[str, int]:
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
        }

    def prefill(self, prompt: str) -> Dict[str, object]:
        request_id = f"req_{torch.randint(0, 2**31 - 1, (1,), device='cpu').item()}"
        inputs = self._tokenize_prompt(prompt)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs, use_cache=True)
            logits = outputs.logits[:, -1, :]
            first_token_id = int(torch.argmax(logits, dim=-1).item())

        initial_kv: List[Dict[str, torch.Tensor]] = []
        for layer_kv in outputs.past_key_values:
            key = layer_kv.key_cache if hasattr(layer_kv, "key_cache") else layer_kv[0]
            value = layer_kv.value_cache if hasattr(layer_kv, "value_cache") else layer_kv[1]
            initial_kv.append(
                {
                    "key": self._normalize_key_cache(key),
                    "value": self._normalize_value_cache(value),
                }
            )

        return {
            "request_id": request_id,
            "prompt_len": int(model_inputs["input_ids"].shape[1]),
            "first_token_id": first_token_id,
            "initial_kv": initial_kv,
        }

    def _tokenize_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        if self.model_type == "qwen":
            token_ids = self.tokenizer.tokenizer.encode(prompt, allowed_special="all")
            if not token_ids:
                raise ValueError("Prompt produced no tokens for Qwen tokenizer")
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        return self.tokenizer(prompt, return_tensors="pt")

    def _normalize_key_cache(self, key: torch.Tensor) -> torch.Tensor:
        if self.model_type == "opt":
            return key[0].permute(1, 0, 2).contiguous().cpu()
        if self.model_type == "qwen":
            return key[0].contiguous().cpu()
        raise ValueError(f"Unsupported model type for cache normalization: {self.model_type}")

    def _normalize_value_cache(self, value: torch.Tensor) -> torch.Tensor:
        if self.model_type == "opt":
            return value[0].permute(1, 0, 2).contiguous().cpu()
        if self.model_type == "qwen":
            return value[0].contiguous().cpu()
        raise ValueError(f"Unsupported model type for cache normalization: {self.model_type}")

    def start_token(self, token_id: int, position: int) -> torch.Tensor:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        if self.model_type == "opt":
            position_ids = torch.tensor([[position]], dtype=torch.long, device=self.device)
            attention_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
            hidden = self.backbone.embed_tokens(input_ids)
            hidden = hidden + self.backbone.embed_positions(attention_mask, position_ids=position_ids)
            return hidden.detach().cpu()

        if self.model_type == "qwen":
            hidden = self.backbone.wte(input_ids)
            hidden = self.backbone.drop(hidden)
            return hidden.detach().cpu()

        raise ValueError(f"Unsupported model type for start_token: {self.model_type}")

    def prepare_attention(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        request_id: str,
        context_len: int,
    ) -> Dict[str, object]:
        del request_id
        hidden = hidden_state.to(self.device)

        if self.model_type == "opt":
            return self._prepare_opt_attention(hidden, layer_idx)
        if self.model_type == "qwen":
            return self._prepare_qwen_attention(hidden, layer_idx, context_len)
        raise ValueError(f"Unsupported model type for prepare_attention: {self.model_type}")

    def _prepare_opt_attention(self, hidden: torch.Tensor, layer_idx: int) -> Dict[str, object]:
        layer = self.layers[layer_idx]
        residual = hidden

        if layer.do_layer_norm_before:
            hidden = layer.self_attn_layer_norm(hidden)

        attn = layer.self_attn
        batch, seq_len, _ = hidden.shape
        if batch != 1 or seq_len != 1:
            raise ValueError("The first refactor supports batch=1, seq_len=1 decode only")

        query = attn.q_proj(hidden) * attn.scaling
        key = attn.k_proj(hidden)
        value = attn.v_proj(hidden)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)

        return {
            "residual": residual.detach().cpu(),
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
            "score_scale": 1.0,
        }

    def _prepare_qwen_attention(self, hidden: torch.Tensor, layer_idx: int, context_len: int) -> Dict[str, object]:
        layer = self.layers[layer_idx]
        residual = hidden
        layernorm_output = layer.ln_1(hidden)
        attn = layer.attn

        mixed_x_layer = attn.c_attn(layernorm_output)
        query, key, value = mixed_x_layer.split(attn.split_size, dim=2)

        query = attn._split_heads(query, self.num_heads, self.head_dim)
        key = attn._split_heads(key, self.num_heads, self.head_dim)
        value = attn._split_heads(value, self.num_heads, self.head_dim)

        rotary_pos_emb = self._build_qwen_rotary_pos_emb(context_len)
        query = self.qwen_module.apply_rotary_pos_emb(query, rotary_pos_emb)
        key = self.qwen_module.apply_rotary_pos_emb(key, rotary_pos_emb)

        if context_len > attn.seq_length and attn.use_logn_attn and not self.model.training:
            seq_start = context_len - query.size(1)
            seq_end = context_len
            logn_tensor = attn.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
            query = query * logn_tensor.expand_as(query)

        return {
            "residual": residual.detach().cpu(),
            "query": query.squeeze(1).detach().cpu(),
            "key": key.squeeze(1).detach().cpu(),
            "value": value.squeeze(1).detach().cpu(),
            "score_scale": 1.0 / math.sqrt(self.head_dim),
        }

    def _build_qwen_rotary_pos_emb(self, context_len: int):
        ntk_alpha = 1.0
        if self.backbone.use_dynamic_ntk and context_len > self.backbone.seq_length:
            ntk_alpha = self.backbone.get_ntk_alpha(context_len)
        rotary_pos_emb = self.backbone.rotary_emb(context_len, ntk_alpha=ntk_alpha)
        return [value[:, -1:, :, :] for value in rotary_pos_emb]

    def finish_layer(self, residual: torch.Tensor, attention_context: torch.Tensor, layer_idx: int) -> torch.Tensor:
        residual = residual.to(self.device)
        context = attention_context.to(self.device)
        if context.dim() == 3:
            context = context.unsqueeze(1)
        attn_output = context.reshape(1, 1, self.hidden_size).contiguous()

        if self.model_type == "opt":
            return self._finish_opt_layer(residual, attn_output, layer_idx)
        if self.model_type == "qwen":
            return self._finish_qwen_layer(residual, attn_output, layer_idx)
        raise ValueError(f"Unsupported model type for finish_layer: {self.model_type}")

    def _finish_opt_layer(self, residual: torch.Tensor, attn_output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        hidden = layer.self_attn.out_proj(attn_output)
        hidden = residual + hidden

        if not layer.do_layer_norm_before:
            hidden = layer.self_attn_layer_norm(hidden)

        hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden.size(-1))
        residual_ffn = hidden

        if layer.do_layer_norm_before:
            hidden = layer.final_layer_norm(hidden)

        hidden = layer.fc1(hidden)
        hidden = layer.activation_fn(hidden)
        hidden = layer.fc2(hidden)
        hidden = residual_ffn + hidden
        hidden = hidden.view(hidden_shape)

        if not layer.do_layer_norm_before:
            hidden = layer.final_layer_norm(hidden)

        return hidden.detach().cpu()

    def _finish_qwen_layer(self, residual: torch.Tensor, attn_output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        attn_output = layer.attn.c_proj(attn_output)
        layernorm_input = attn_output + residual
        layernorm_output = layer.ln_2(layernorm_input)
        hidden = layernorm_input + layer.mlp(layernorm_output)
        return hidden.detach().cpu()

    def sample_next_token(self, hidden_state: torch.Tensor) -> int:
        hidden = hidden_state.to(self.device)
        if self.model_type == "opt":
            final_norm = getattr(self.backbone, "final_layer_norm", None)
            if final_norm is not None:
                hidden = final_norm(hidden)
        elif self.model_type == "qwen":
            hidden = self.backbone.ln_f(hidden)
        else:
            raise ValueError(f"Unsupported model type for sample_next_token: {self.model_type}")

        logits = self.model.lm_head(hidden)
        return int(torch.argmax(logits[:, -1, :], dim=-1).item())

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _patch_qwen_runtime_compatibility(self, model):
        backbone = getattr(model, "transformer", None)
        if backbone is None or hasattr(backbone, "get_head_mask"):
            return

        def _get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers

            head_mask = head_mask.to(dtype=this.dtype)
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported head_mask dim: {head_mask.dim()}")

            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        backbone.get_head_mask = _get_head_mask.__get__(backbone, type(backbone))
