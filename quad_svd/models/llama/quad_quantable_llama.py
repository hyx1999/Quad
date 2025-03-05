import functools
import quad
import quad.modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    LlamaMLP, 
    LlamaForCausalLM, 
    StaticCache,
    repeat_kv,
    apply_rotary_pos_emb,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache
from quad import TensorPack
from quad.modules import QuantLinearFp16, Identity, OnlineHadamard
from torch import Tensor
from types import MethodType
import torch.nn.functional as F

ALL_LAYERNORM_LAYERS.append(quad.modules.RMSNorm)


def add_lora_in_linear(module):
    def get_fwd_fn():
        def lora_forward(self, input: Tensor) -> Tensor:
            lora_A = self.adapters["lora_A"]
            lora_B = self.adapters["lora_B"]
            return lora_A(lora_B(input))
        return lora_forward
    if hasattr(module, "adapters"):
        fwd_fn = get_fwd_fn()
        setattr(module, "lora_forward", MethodType(fwd_fn, module))

class QuadQuantableLlamaConfig(LlamaConfig):
    model_type = "quad_quantable_llama"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_rank = kwargs.get("pod_rank", 0)
        self.svd_rank = kwargs.get("svd_rank", 0)
    

class QuadQuantableLlamaAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config: QuadQuantableLlamaConfig = self.config
        self.q_proj = nn.Linear(self.hidden_size + config.pod_rank, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size + config.pod_rank, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size + config.pod_rank, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size + config.pod_rank, bias=config.attention_bias)
        if config.svd_rank > 0:
            self.o_proj.adapters = nn.ModuleDict(
                {
                    "lora_A": nn.Linear(
                        out_features=self.hidden_size + config.pod_rank,
                        in_features=config.svd_rank,
                        bias=False,
                    ),
                    "lora_B": nn.Linear(
                        out_features=config.svd_rank,
                        in_features=self.num_heads * self.head_dim,
                        bias=False,
                    ),
                }
            )
            add_lora_in_linear(self.o_proj)
        else:
            self.o_proj.register_module("adapters", None)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output) + self.o_proj.lora_forward(attn_output)

        return attn_output, None, past_key_value


class QuadQuantableLlamaMLP(LlamaMLP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config: QuadQuantableLlamaConfig = self.config
        self.gate_proj = nn.Linear(self.hidden_size + config.pod_rank, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size + config.pod_rank, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size + config.pod_rank, bias=config.mlp_bias)
        if config.svd_rank > 0:
            self.down_proj.adapters = nn.ModuleDict(
                {
                    "lora_A": nn.Linear(
                        out_features=self.hidden_size + config.pod_rank,
                        in_features=config.svd_rank,
                        bias=False,
                    ),
                    "lora_B": nn.Linear(
                        out_features=config.svd_rank,
                        in_features=self.num_heads * self.head_dim,
                        bias=False,
                    ),
                }
            )
            add_lora_in_linear(self.o_proj)
        else:
            self.down_proj.register_module("adapters", None)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise ValueError
        else:
            intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(intermediate) + self.down_proj.lora_forward(intermediate)
        return down_proj

class QuadQuantableLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self._expand_embedding()
        self._expand_lm_head()
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuadQuantableLlamaAttention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuadQuantableLlamaMLP(config=config)
        self.model.norm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _expand_embedding(self):
        embed_tokens: nn.Embedding = self.model.embed_tokens
        embed_tokens.embedding_dim += self.config.pod_rank
        embed_tokens.weight.data = torch.empty(
            (embed_tokens.num_embeddings, embed_tokens.embedding_dim),
            dtype=embed_tokens.weight.dtype,
            device=embed_tokens.weight.device,
        )
    
    def _expand_lm_head(self):
        lm_head: nn.Linear = self.lm_head
        lm_head.in_features += self.config.pod_rank
        lm_head.weight.data = torch.empty(
            (lm_head.out_features, lm_head.in_features),
            dtype=lm_head.weight.dtype,
            device=lm_head.weight.device,
        )
