"""Encoder-decoder Transformer with incremental decoding and architecture switches."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, kv_heads=None, rope=False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads or heads
        self.head_dim = dim // heads
        if dim % heads or heads % self.kv_heads:
            raise ValueError("d_model must divide into heads, and heads into kv_heads.")
        if rope and self.head_dim % 2:
            raise ValueError("RoPE requires even head_dim.")
        self.rope = rope
        self.dropout = dropout
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.v = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.out = nn.Linear(dim, dim)
        self.register_buffer(
            "frequencies",
            10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim),
            persistent=False,
        )

    def rotate(self, x, positions):
        if not self.rope:
            return x
        angles = positions.float()[:, None] * self.frequencies[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), -1
        ).flatten(-2)

    def queries(self, x, positions):
        b, t, _ = x.shape
        return self.rotate(
            self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2), positions
        )

    def keys_values(self, x, positions):
        b, t, _ = x.shape
        k = self.k(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        return self.rotate(k, positions), v

    def attend(self, q, k, v, allowed=None):
        # Portable GQA fallback for torch 2.2. Benchmark before claiming a speedup.
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(result.transpose(1, 2).contiguous().flatten(2))

    def forward(self, x, memory, allowed, positions):
        k, v = self.keys_values(memory, positions)
        return self.attend(self.queries(x, positions), k, v, allowed)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, activation, dropout):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            hidden = math.ceil((hidden * 2 / 3) / 64) * 64
        self.up = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden) if activation == "swiglu" else None
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.up(x)
        if self.activation == "swiglu":
            hidden = F.silu(hidden) * self.gate(x)
        elif self.activation == "gelu":
            hidden = F.gelu(hidden)
        else:
            hidden = F.relu(hidden)
        return self.down(self.dropout(hidden))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ffn_dim,
        dropout,
        norm_type,
        norm_style,
        activation,
        rope,
        decoder=False,
        kv_heads=None,
    ):
        super().__init__()
        self.pre_norm = norm_style == "pre"
        self.self_attention = Attention(
            dim, heads, dropout, kv_heads if decoder else heads, rope
        )
        self.cross_attention = Attention(dim, heads, dropout) if decoder else None
        self.ffn = FeedForward(dim, ffn_dim, activation, dropout)
        norm = RMSNorm if norm_type == "rms" else nn.LayerNorm
        self.norms = nn.ModuleList([norm(dim) for _ in range(3 if decoder else 2)])
        self.dropout = nn.Dropout(dropout)

    def input_norm(self, x, idx):
        return self.norms[idx](x) if self.pre_norm else x

    def residual(self, x, update, idx):
        x = x + self.dropout(update)
        return x if self.pre_norm else self.norms[idx](x)

    def forward(self, x, allowed, positions, memory=None, memory_allowed=None):
        normalized = self.input_norm(x, 0)
        x = self.residual(
            x, self.self_attention(normalized, normalized, allowed, positions), 0
        )
        idx = 1
        if memory is not None:
            # Cross-attention uses no rotary positions: source/target coordinates differ.
            k, v = self.cross_attention.keys_values(memory, positions)
            q = self.cross_attention.queries(self.input_norm(x, 1), positions)
            x = self.residual(
                x, self.cross_attention.attend(q, k, v, memory_allowed), 1
            )
            idx = 2
        return self.residual(x, self.ffn(self.input_norm(x, idx)), idx)

    def step(self, x, state, position, memory_allowed, cache_type):
        positions = position.reshape(1)
        normalized = self.input_norm(x, 0)
        q = self.self_attention.queries(normalized, positions)
        k, v = self.self_attention.keys_values(normalized, positions)
        if cache_type == "static":
            state["k"].index_copy_(2, positions, k)
            state["v"].index_copy_(2, positions, v)
            k, v = state["k"], state["v"]
            allowed = state["positions"] <= position
        else:
            k = k if state["k"] is None else torch.cat((state["k"], k), dim=2)
            v = v if state["v"] is None else torch.cat((state["v"], v), dim=2)
            state["k"], state["v"] = k, v
            allowed = None
        x = self.residual(x, self.self_attention.attend(q, k, v, allowed), 0)
        q = self.cross_attention.queries(self.input_norm(x, 1), positions)
        x = self.residual(
            x,
            self.cross_attention.attend(
                q, state["cross_k"], state["cross_v"], memory_allowed
            ),
            1,
        )
        return self.residual(x, self.ffn(self.input_norm(x, 2)), 2)


class CodeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        encoder_layers=4,
        decoder_layers=4,
        ffn_dim=1280,
        dropout=0.1,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        max_source_length=256,
        max_target_length=256,
        norm_type="layer",
        norm_style="post",
        activation="relu",
        position_encoding="learned",
        kv_heads=None,
        tie_output=False,
        layer_repeats=1,
    ):
        super().__init__()
        if norm_type not in {"layer", "rms"} or norm_style not in {"pre", "post"}:
            raise ValueError("Invalid normalization type/style.")
        if activation not in {"relu", "gelu", "swiglu"}:
            raise ValueError("Invalid FFN activation.")
        if position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be learned or rope.")
        if (
            isinstance(layer_repeats, bool)
            or not isinstance(layer_repeats, int)
            or layer_repeats < 1
        ):
            raise ValueError("layer_repeats must be a positive integer.")
        self.layer_repeats = layer_repeats
        if (
            min(
                vocab_size,
                d_model,
                n_heads,
                encoder_layers,
                decoder_layers,
                ffn_dim,
                max_source_length,
                max_target_length,
            )
            <= 0
        ):
            raise ValueError("Model dimensions must be positive.")
        if (
            len({pad_id, bos_id, eos_id}) != 3
            or min(pad_id, bos_id, eos_id) < 0
            or max(pad_id, bos_id, eos_id) >= vocab_size
        ):
            raise ValueError(
                "Special token IDs must be distinct and inside the vocabulary."
            )
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.max_source_length, self.max_target_length = (
            max_source_length,
            max_target_length,
        )
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positions = (
            nn.Embedding(max(max_source_length, max_target_length), d_model)
            if position_encoding == "learned"
            else None
        )
        args = (
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            norm_type,
            norm_style,
            activation,
            position_encoding == "rope",
        )
        self.encoder = nn.ModuleList([Block(*args) for _ in range(encoder_layers)])
        self.decoder = nn.ModuleList(
            [
                Block(*args, decoder=True, kv_heads=kv_heads)
                for _ in range(decoder_layers)
            ]
        )
        norm = RMSNorm if norm_type == "rms" else nn.LayerNorm
        self.encoder_norm = norm(d_model) if norm_style == "pre" else nn.Identity()
        self.decoder_norm = norm(d_model) if norm_style == "pre" else nn.Identity()
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._initialize)
        if tie_output:
            self.output.weight = self.embedding.weight
        self._compiled_step = None

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def embed(self, ids, positions):
        x = self.embedding(ids) * math.sqrt(self.d_model)
        if self.positions is not None:
            x = x + self.positions(positions)
        return self.dropout(x)

    def encode(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.max_source_length:
            raise ValueError("Source sequence exceeds model position limit.")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        valid = (
            input_ids.ne(self.pad_id)
            if attention_mask is None
            else attention_mask.bool()
        )
        allowed = valid[:, None, None, :]
        x = self.embed(input_ids, positions)
        for layer in self.encoder:
            for _ in range(self.layer_repeats):
                x = layer(x, allowed, positions)
        return self.encoder_norm(x), allowed

    def decode(self, decoder_input_ids, memory, memory_allowed):
        t = decoder_input_ids.shape[1]
        if t > self.max_target_length:
            raise ValueError("Target sequence exceeds model position limit.")
        positions = torch.arange(t, device=decoder_input_ids.device)
        causal = positions[:, None] >= positions[None, :]
        allowed = (
            causal[None, None, :, :]
            & decoder_input_ids.ne(self.pad_id)[:, None, None, :]
        )
        x = self.embed(decoder_input_ids, positions)
        for layer in self.decoder:
            for _ in range(self.layer_repeats):
                x = layer(x, allowed, positions, memory, memory_allowed)
        return self.output(self.decoder_norm(x))

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, **unused):
        memory, allowed = self.encode(input_ids, attention_mask)
        return {"logits": self.decode(decoder_input_ids, memory, allowed)}

    def init_cache(self, memory, max_length, cache_type="static"):
        if cache_type not in {"static", "dynamic"}:
            raise ValueError("cache_type must be static or dynamic.")
        if not 0 < max_length <= self.max_target_length:
            raise ValueError("Generation length exceeds model position limit.")
        states = []
        for layer in self.decoder:
            k, v = layer.cross_attention.keys_values(memory, None)
            shape = (
                memory.shape[0],
                layer.self_attention.kv_heads,
                max_length,
                layer.self_attention.head_dim,
            )
            # Cross K/V depend only on the shared projection and encoder memory.
            # Self K/V depend on the execution occurrence, even with tied weights.
            for _ in range(self.layer_repeats):
                states.append(
                    {
                        "k": k.new_zeros(shape) if cache_type == "static" else None,
                        "v": v.new_zeros(shape) if cache_type == "static" else None,
                        "cross_k": k,
                        "cross_v": v,
                        "positions": torch.arange(max_length, device=memory.device),
                    }
                )
        return states

    def decode_step(
        self, last_token, states, position, memory_allowed, cache_type="static"
    ):
        if len(states) != len(self.decoder) * self.layer_repeats:
            raise ValueError(
                "Decoder cache must contain one state per layer execution."
            )
        x = self.embed(last_token[:, None], position.reshape(1))
        for index, layer in enumerate(self.decoder):
            for repeat in range(self.layer_repeats):
                state = states[index * self.layer_repeats + repeat]
                x = layer.step(x, state, position, memory_allowed, cache_type)
        return self.output(self.decoder_norm(x[:, 0]))

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=64,
        cache_type="static",
        compile_step=False,
        eos_check_interval=1,
        num_beams=1,
        length_penalty=0.6,
        suppress_tokens=(),
    ):
        if self.training:
            raise RuntimeError("Call model.eval() before cached generation.")
        if eos_check_interval < 1:
            raise ValueError("eos_check_interval must be positive.")
        if compile_step and cache_type != "static":
            raise ValueError("Compiled decoding requires static cache shapes.")
        if num_beams < 1:
            raise ValueError("num_beams must be positive.")
        if num_beams > 1:
            if compile_step:
                raise ValueError(
                    "Compiled step currently supports greedy decoding only."
                )
            return self.beam_generate(
                input_ids,
                attention_mask,
                max_new_tokens,
                cache_type,
                num_beams,
                length_penalty,
                suppress_tokens,
            )
        memory, allowed = self.encode(input_ids, attention_mask)
        states = self.init_cache(memory, max_new_tokens, cache_type)
        if compile_step and self._compiled_step is None:
            self._compiled_step = torch.compile(
                self.decode_step, mode="reduce-overhead"
            )
        step_fn = self._compiled_step if compile_step else self.decode_step
        last = input_ids.new_full((len(input_ids),), self.bos_id)
        result = input_ids.new_full((len(input_ids), max_new_tokens), self.pad_id)
        finished = torch.zeros(
            len(input_ids), dtype=torch.bool, device=input_ids.device
        )
        positions = torch.arange(max_new_tokens, device=input_ids.device)
        for step in range(max_new_tokens):
            logits = step_fn(last, states, positions[step], allowed, cache_type)
            # PAD/BOS are structural tokens, not valid code continuations.
            logits[:, self.pad_id] = -torch.inf
            logits[:, self.bos_id] = -torch.inf
            for token_id in suppress_tokens:
                logits[:, token_id] = -torch.inf
            token = logits.argmax(-1).masked_fill(finished, self.pad_id)
            result[:, step] = token
            finished = finished | token.eq(self.eos_id)
            last = token
            if (step + 1) % eos_check_interval == 0 and bool(finished.all()):
                break
        return result[:, : step + 1]

    def beam_generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        cache_type,
        beams,
        alpha,
        suppress_tokens,
    ):
        memory, allowed = self.encode(input_ids, attention_mask)
        batch = len(input_ids)
        memory, allowed = memory.repeat_interleave(beams, 0), allowed.repeat_interleave(
            beams, 0
        )
        states = self.init_cache(memory, max_length, cache_type)
        last = input_ids.new_full((batch * beams,), self.bos_id)
        result = input_ids.new_full((batch, beams, max_length), self.pad_id)
        scores = memory.new_full((batch, beams), -torch.inf, dtype=torch.float32)
        scores[:, 0] = 0
        finished = torch.zeros((batch, beams), device=memory.device, dtype=torch.bool)
        lengths = torch.zeros((batch, beams), device=memory.device, dtype=torch.long)
        positions = torch.arange(max_length, device=memory.device)
        offsets = torch.arange(batch, device=memory.device)[:, None] * beams
        for step in range(max_length):
            logits = self.decode_step(
                last, states, positions[step], allowed, cache_type
            ).float()
            logits[:, self.pad_id] = -torch.inf
            logits[:, self.bos_id] = -torch.inf
            for token_id in suppress_tokens:
                logits[:, token_id] = -torch.inf
            log_probs = logits.log_softmax(-1).view(batch, beams, -1)
            log_probs = log_probs.masked_fill(finished[:, :, None], -torch.inf)
            log_probs[:, :, self.pad_id] = torch.where(
                finished, 0.0, log_probs[:, :, self.pad_id]
            )
            vocab = log_probs.shape[-1]
            scores, selected = (
                (scores[:, :, None] + log_probs).flatten(1).topk(beams, dim=1)
            )
            parent, token = selected // vocab, selected % vocab
            result = result.gather(1, parent[:, :, None].expand(-1, -1, max_length))
            result[:, :, step] = token
            was_finished = finished.gather(1, parent)
            lengths = lengths.gather(1, parent) + (~was_finished).long()
            finished = was_finished | token.eq(self.eos_id)
            indices = (parent + offsets).flatten()
            for state in states:
                for name in ["k", "v", "cross_k", "cross_v"]:
                    state[name] = state[name].index_select(0, indices)
            last = token.flatten()
            if bool(finished.all()):
                break
        normalized = scores / ((5 + lengths.float()) / 6).pow(alpha)
        best = normalized.argmax(-1)
        return result[torch.arange(batch, device=memory.device), best, : step + 1]
