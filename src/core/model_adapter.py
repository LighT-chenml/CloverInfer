from __future__ import annotations

import importlib
import math
import time
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
        # GQA/MQA models (e.g. newer Qwen/Llama variants) may expose a smaller
        # KV head count. If absent, default to standard MHA behavior.
        self.num_kv_heads = int(getattr(self.model.config, "num_key_value_heads", self.num_heads) or self.num_heads)
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
        elif self.model_type == "llama":
            # LlamaForCausalLM: model.model is the decoder backbone.
            backbone = getattr(self.model, "model", None)
            if backbone is None:
                raise ValueError("Unsupported Llama model wrapper: missing `.model` attribute")
            self.backbone = backbone
            self.layers = getattr(self.backbone, "layers", None)
            if self.layers is None:
                raise ValueError("Unsupported Llama backbone: missing `.layers` attribute")
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
            "model_type": str(self.model_type),
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": int(self.head_dim),
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
        if self.model_type == "llama":
            # Llama caches can appear as either:
            # - [B, heads, T, D] (common)
            # - [B, T, heads, D] (some implementations)
            # Normalize to [T, heads, D] on CPU.
            t = key.detach()
            if t.dim() == 4:
                t = t[0]
            if t.dim() != 3:
                raise ValueError(f"Unsupported Llama key cache shape: {tuple(t.shape)}")
            # If the first dim looks like heads, transpose to [T, heads, D].
            if int(t.shape[0]) in {int(self.num_heads), int(self.num_kv_heads)}:
                return t.permute(1, 0, 2).contiguous().cpu()
            return t.contiguous().cpu()
        raise ValueError(f"Unsupported model type for cache normalization: {self.model_type}")

    def _normalize_value_cache(self, value: torch.Tensor) -> torch.Tensor:
        if self.model_type == "opt":
            return value[0].permute(1, 0, 2).contiguous().cpu()
        if self.model_type == "qwen":
            return value[0].contiguous().cpu()
        if self.model_type == "llama":
            t = value.detach()
            if t.dim() == 4:
                t = t[0]
            if t.dim() != 3:
                raise ValueError(f"Unsupported Llama value cache shape: {tuple(t.shape)}")
            if int(t.shape[0]) in {int(self.num_heads), int(self.num_kv_heads)}:
                return t.permute(1, 0, 2).contiguous().cpu()
            return t.contiguous().cpu()
        raise ValueError(f"Unsupported model type for cache normalization: {self.model_type}")

    def _denormalize_past_key_values(self, initial_kv: List[Dict[str, torch.Tensor]]):
        past_key_values = []
        for layer_kv in initial_kv:
            key = layer_kv["key"].to(self.device)
            value = layer_kv["value"].to(self.device)
            if self.model_type == "opt":
                key = key.permute(1, 0, 2).unsqueeze(0).contiguous()
                value = value.permute(1, 0, 2).unsqueeze(0).contiguous()
            elif self.model_type == "qwen":
                key = key.unsqueeze(0).contiguous()
                value = value.unsqueeze(0).contiguous()
            elif self.model_type == "llama":
                # Transformers v5 prefers a Cache instance. Build a DynamicCache
                # from the layer tensors (expected per-layer shape [B, heads, T, D]).
                # Here `key/value` are [T, heads, D], so convert accordingly.
                from transformers.cache_utils import DynamicCache

                ddp_cache_data = []
                for item in initial_kv:
                    k = item["key"].to(self.device)
                    v = item["value"].to(self.device)
                    # [T, heads, D] -> [B, heads, T, D]
                    k = k.permute(1, 0, 2).unsqueeze(0).contiguous()
                    v = v.permute(1, 0, 2).unsqueeze(0).contiguous()
                    ddp_cache_data.append((k, v))
                return DynamicCache(ddp_cache_data=ddp_cache_data, config=self.model.config)
            else:
                raise ValueError(f"Unsupported model type for cache restore: {self.model_type}")
            past_key_values.append((key, value))
        return tuple(past_key_values)

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

        if self.model_type == "llama":
            # Llama uses RoPE inside attention; no explicit position embedding here.
            hidden = self.backbone.embed_tokens(input_ids)
            return hidden.detach().cpu()

        raise ValueError(f"Unsupported model type for start_token: {self.model_type}")

    def start_token_batch(self, token_ids: list[int], positions: list[int]) -> torch.Tensor:
        if len(token_ids) != len(positions):
            raise ValueError("token_ids and positions must have the same length")
        if not token_ids:
            return torch.empty((0, 1, self.hidden_size), dtype=self.dtype).cpu()

        input_ids = torch.tensor([[int(tid)] for tid in token_ids], dtype=torch.long, device=self.device)

        if self.model_type == "opt":
            position_ids = torch.tensor([[int(pos)] for pos in positions], dtype=torch.long, device=self.device)
            attention_mask = torch.ones((len(token_ids), 1), dtype=torch.long, device=self.device)
            hidden = self.backbone.embed_tokens(input_ids)
            hidden = hidden + self.backbone.embed_positions(attention_mask, position_ids=position_ids)
            return hidden.detach().cpu()

        if self.model_type == "qwen":
            hidden = self.backbone.wte(input_ids)
            hidden = self.backbone.drop(hidden)
            return hidden.detach().cpu()

        if self.model_type == "llama":
            hidden = self.backbone.embed_tokens(input_ids)
            return hidden.detach().cpu()

        raise ValueError(f"Unsupported model type for start_token_batch: {self.model_type}")

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
        if self.model_type == "llama":
            return self._prepare_llama_attention(hidden, layer_idx, context_len)
        raise ValueError(f"Unsupported model type for prepare_attention: {self.model_type}")

    def prepare_attention_batch(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        request_ids: list[str],
        context_lens: list[int],
    ) -> Dict[str, object]:
        del request_ids
        if hidden_state.dim() != 3:
            raise ValueError("hidden_state must be [B, 1, H]")
        batch = int(hidden_state.shape[0])
        if batch != len(context_lens):
            raise ValueError("context_lens length must match hidden_state batch size")
        hidden = hidden_state.to(self.device)

        if self.model_type == "opt":
            return self._prepare_opt_attention_batch(hidden, layer_idx)
        if self.model_type == "qwen":
            return self._prepare_qwen_attention_batch(hidden, layer_idx, context_lens)
        if self.model_type == "llama":
            return self._prepare_llama_attention_batch(hidden, layer_idx, context_lens)
        raise ValueError(f"Unsupported model type for prepare_attention_batch: {self.model_type}")

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

    def _prepare_opt_attention_batch(self, hidden: torch.Tensor, layer_idx: int) -> Dict[str, object]:
        layer = self.layers[layer_idx]
        residual = hidden

        if layer.do_layer_norm_before:
            hidden = layer.self_attn_layer_norm(hidden)

        attn = layer.self_attn
        batch, seq_len, _ = hidden.shape
        if seq_len != 1:
            raise ValueError("decode expects seq_len=1 for prepare_attention_batch")

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

    def _prepare_qwen_attention_batch(
        self, hidden: torch.Tensor, layer_idx: int, context_lens: list[int]
    ) -> Dict[str, object]:
        batch = int(hidden.shape[0])
        residuals: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        score_scale = 1.0 / math.sqrt(self.head_dim)
        for i in range(batch):
            prepared = self._prepare_qwen_attention(hidden[i : i + 1], layer_idx, int(context_lens[i]))
            residuals.append(prepared["residual"])
            queries.append(prepared["query"])
            keys.append(prepared["key"])
            values.append(prepared["value"])
        return {
            "residual": torch.cat(residuals, dim=0),
            "query": torch.stack(queries, dim=0),
            "key": torch.stack(keys, dim=0),
            "value": torch.stack(values, dim=0),
            "score_scale": float(score_scale),
        }

    def _build_qwen_rotary_pos_emb(self, context_len: int):
        ntk_alpha = 1.0
        if self.backbone.use_dynamic_ntk and context_len > self.backbone.seq_length:
            ntk_alpha = self.backbone.get_ntk_alpha(context_len)
        rotary_pos_emb = self.backbone.rotary_emb(context_len, ntk_alpha=ntk_alpha)
        return [value[:, -1:, :, :] for value in rotary_pos_emb]

    def _prepare_llama_attention(self, hidden: torch.Tensor, layer_idx: int, context_len: int) -> Dict[str, object]:
        # Mirrors LlamaDecoderLayer/LlamaAttention qkv + RoPE, but leaves the
        # actual attention (QK/softmax/AV) to the disaggregated attention node.
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        layer = self.layers[layer_idx]
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        attn = layer.self_attn
        batch, seq_len, _ = hidden.shape
        if batch != 1 or seq_len != 1:
            raise ValueError("The first refactor supports batch=1, seq_len=1 decode only")

        # Position id for the *new* token is the current cache length.
        position_ids = torch.tensor([[int(context_len)]], dtype=torch.long, device=self.device)
        cos, sin = self.backbone.rotary_emb(hidden, position_ids=position_ids)

        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, int(attn.head_dim))
        query = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)  # [B, q_heads, 1, D]
        key = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)    # [B, kv_heads, 1, D]
        value = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)  # [B, kv_heads, 1, D]

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Expand KV heads to attention heads if the model uses GQA/MQA so that
        # the downstream attention backend can stay in the "heads match" regime.
        key = repeat_kv(key, int(attn.num_key_value_groups))
        value = repeat_kv(value, int(attn.num_key_value_groups))

        return {
            "residual": residual.detach().cpu(),
            "query": query[:, :, -1, :].squeeze(0).detach().cpu(),
            "key": key[:, :, -1, :].squeeze(0).detach().cpu(),
            "value": value[:, :, -1, :].squeeze(0).detach().cpu(),
            "score_scale": float(attn.scaling),
        }

    def _prepare_llama_attention_batch(self, hidden: torch.Tensor, layer_idx: int, context_lens: list[int]) -> Dict[str, object]:
        # Keep it simple and robust: per-sample RoPE depends on position, so loop.
        batch = int(hidden.shape[0])
        residuals: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        score_scale = 1.0 / math.sqrt(self.head_dim)
        for i in range(batch):
            prepared = self._prepare_llama_attention(hidden[i : i + 1], layer_idx, int(context_lens[i]))
            residuals.append(prepared["residual"])
            queries.append(prepared["query"])
            keys.append(prepared["key"])
            values.append(prepared["value"])
            score_scale = float(prepared.get("score_scale", score_scale))
        return {
            "residual": torch.cat(residuals, dim=0),
            "query": torch.stack(queries, dim=0),
            "key": torch.stack(keys, dim=0),
            "value": torch.stack(values, dim=0),
            "score_scale": float(score_scale),
        }

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
        if self.model_type == "llama":
            return self._finish_llama_layer(residual, attn_output, layer_idx)
        raise ValueError(f"Unsupported model type for finish_layer: {self.model_type}")

    def finish_layer_batch(
        self,
        residual: torch.Tensor,
        attention_context: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if residual.dim() != 3:
            raise ValueError("residual must be [B, 1, H]")
        context = attention_context
        if context.dim() == 3:
            context = context.unsqueeze(1)
        if context.dim() != 4:
            raise ValueError("attention_context must be [B, heads, dim] or [B, 1, heads, dim]")
        if residual.shape[0] != context.shape[0]:
            raise ValueError("residual and attention_context batch sizes must match")

        residual_dev = residual.to(self.device)
        attn_output = context.reshape(residual_dev.shape[0], 1, self.hidden_size).contiguous().to(self.device)
        if self.model_type == "opt":
            return self._finish_opt_layer(residual_dev, attn_output, layer_idx)
        if self.model_type == "qwen":
            return self._finish_qwen_layer(residual_dev, attn_output, layer_idx)
        if self.model_type == "llama":
            return self._finish_llama_layer(residual_dev, attn_output, layer_idx)
        raise ValueError(f"Unsupported model type for finish_layer_batch: {self.model_type}")

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

    def _finish_llama_layer(self, residual: torch.Tensor, attn_output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        attn_out = layer.self_attn.o_proj(attn_output)
        hidden = residual + attn_out
        residual_ffn = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual_ffn + hidden
        return hidden.detach().cpu()

    def sample_next_token(self, hidden_state: torch.Tensor) -> int:
        hidden = hidden_state.to(self.device)
        if self.model_type == "opt":
            final_norm = getattr(self.backbone, "final_layer_norm", None)
            if final_norm is not None:
                hidden = final_norm(hidden)
        elif self.model_type == "qwen":
            hidden = self.backbone.ln_f(hidden)
        elif self.model_type == "llama":
            hidden = self.backbone.norm(hidden)
        else:
            raise ValueError(f"Unsupported model type for sample_next_token: {self.model_type}")

        logits = self.model.lm_head(hidden)
        return int(torch.argmax(logits[:, -1, :], dim=-1).item())

    def sample_next_token_batch(self, hidden_state: torch.Tensor) -> list[int]:
        if hidden_state.dim() != 3:
            raise ValueError("hidden_state must be [B, 1, H]")
        hidden = hidden_state.to(self.device)
        if self.model_type == "opt":
            final_norm = getattr(self.backbone, "final_layer_norm", None)
            if final_norm is not None:
                hidden = final_norm(hidden)
        elif self.model_type == "qwen":
            hidden = self.backbone.ln_f(hidden)
        elif self.model_type == "llama":
            hidden = self.backbone.norm(hidden)
        else:
            raise ValueError(f"Unsupported model type for sample_next_token_batch: {self.model_type}")

        logits = self.model.lm_head(hidden)
        token_ids = torch.argmax(logits[:, -1, :], dim=-1)
        return [int(item) for item in token_ids.detach().cpu().tolist()]

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def greedy_generate(self, prompt: str, max_new_tokens: int) -> Dict[str, object]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        inputs = self._tokenize_prompt(prompt)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        prompt_len = int(model_inputs["input_ids"].shape[1])

        started_at = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**model_inputs, use_cache=True)
            first_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            first_token_at = time.perf_counter()

            generated_ids = [first_token_id]
            current_token_id = first_token_id
            past_key_values = outputs.past_key_values

            for step in range(1, max_new_tokens):
                total_len = prompt_len + step
                decode_inputs = {
                    "input_ids": torch.tensor([[current_token_id]], dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones((1, total_len), dtype=torch.long, device=self.device),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                outputs = self.model(**decode_inputs)
                current_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                past_key_values = outputs.past_key_values
                generated_ids.append(current_token_id)

        finished_at = time.perf_counter()
        latency = float(finished_at - started_at)
        ttft = float(first_token_at - started_at)
        total_tokens = len(generated_ids)
        tpot = float((latency - ttft) / max(total_tokens - 1, 1))
        throughput = float(total_tokens / latency) if latency > 0 else 0.0

        return {
            "prompt_len": prompt_len,
            "generated_ids": generated_ids,
            "text": self.decode_tokens(generated_ids),
            "metrics": {
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "throughput": throughput,
                "total_tokens": total_tokens,
            },
        }

    def continue_greedy_generate(
        self,
        initial_kv: List[Dict[str, torch.Tensor]],
        prompt_len: int,
        first_token_id: int,
        max_new_tokens: int,
    ) -> Dict[str, object]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        started_at = time.perf_counter()
        generated_ids = [int(first_token_id)]

        if max_new_tokens == 1:
            finished_at = time.perf_counter()
            latency = float(finished_at - started_at)
            return {
                "generated_ids": generated_ids,
                "text": self.decode_tokens(generated_ids),
                "metrics": {
                    "latency": latency,
                    "generated_tokens_after_prefill": 0,
                },
            }

        with torch.no_grad():
            past_key_values = self._denormalize_past_key_values(initial_kv)
            current_token_id = int(first_token_id)

            for step in range(1, max_new_tokens):
                total_len = int(prompt_len) + step
                decode_inputs = {
                    "input_ids": torch.tensor([[current_token_id]], dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones((1, total_len), dtype=torch.long, device=self.device),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                outputs = self.model(**decode_inputs)
                current_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                past_key_values = outputs.past_key_values
                generated_ids.append(current_token_id)

        finished_at = time.perf_counter()
        latency = float(finished_at - started_at)
        return {
            "generated_ids": generated_ids,
            "text": self.decode_tokens(generated_ids),
            "metrics": {
                "latency": latency,
                "generated_tokens_after_prefill": max(0, len(generated_ids) - 1),
            },
        }

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
