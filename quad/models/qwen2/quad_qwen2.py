import functools
import quad
import quad.modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2FlashAttention2, 
    Qwen2MLP, 
    Qwen2ForCausalLM, 
    StaticCache,
    apply_rotary_pos_emb,
    logger,
    _flash_attention_forward
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache
from quad import TensorPack
from quad.modules import QuantLinearW4A4, QuantLinearW4A8, Quantizer, OnlineHadamard
from tqdm import tqdm
from enum import Enum

ALL_LAYERNORM_LAYERS.append(quad.modules.RMSNorm)

class QuantMode(str, Enum):
    w4a4 = "w4a4"
    w4a8 = "w4a8"
    w4a4a8 = "w4a4a8"

class QuadQwen2Config(Qwen2Config):
    model_type = "quad_qwen2"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_rank = kwargs.get("pod_rank", 0)
        self.input_clip_ratio = kwargs.get("input_clip_ratio", 1.0)
        self.quant_mode = kwargs.get("quant_mode", QuantMode.w4a4a8)

class LinearTypeMixin:
    
    def get_linear_type(self):
        if self.config.quant_mode == QuantMode.w4a4:
            QuantLinearU = QuantLinearW4A4
            QuantLinearD = QuantLinearW4A4
        elif self.config.quant_mode == QuantMode.w4a8:
            QuantLinearU = QuantLinearW4A8
            QuantLinearD = QuantLinearW4A8
        else:
            QuantLinearU = QuantLinearW4A4
            QuantLinearD = QuantLinearW4A8
        return QuantLinearU, QuantLinearD
    
    def get_act_type(self):
        if self.config.quant_mode == QuantMode.w4a4:
            ActTypeU = "int4"
            ActTypeD = "int4"
        elif self.config.quant_mode == QuantMode.w4a8:
            ActTypeU = "int8"
            ActTypeD = "int8"
        else:
            ActTypeU = "int4"
            ActTypeD = "int8"
        return ActTypeU, ActTypeD        


class QuadQwen2Attention(Qwen2FlashAttention2, LinearTypeMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  

        QLinearU, QLinearD = self.get_linear_type()
        actU, actD = self.get_act_type()
        
        config: QuadQwen2Config = self.config
        self.quantizer = Quantizer(
            config.hidden_size,
            config.pod_rank,
            config.input_clip_ratio,
            act_dtype=actU
        )
        self.q_proj = QLinearU.from_float(self.q_proj, pod_rank=config.pod_rank)
        self.k_proj = QLinearU.from_float(self.k_proj, pod_rank=config.pod_rank)
        self.v_proj = QLinearU.from_float(self.v_proj, pod_rank=config.pod_rank)
        self.o_proj_hadamard = quad.modules.OnlineHadamard(self.num_heads)
        self.o_proj = nn.Sequential(
            Quantizer(config.hidden_size, 0, config.input_clip_ratio, act_dtype=actD),
            QLinearD.from_float(self.o_proj, extra_out=config.pod_rank)
        )
        
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
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        
        # print("attn_hidden_states:", hidden_states)

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        hidden_states_pack: TensorPack = self.quantizer(hidden_states)
        query_states = self.q_proj(hidden_states_pack)
        key_states = self.k_proj(hidden_states_pack)
        value_states = self.v_proj(hidden_states_pack)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
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

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuadQwen2MLP(Qwen2MLP, LinearTypeMixin):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config = config
        QLinearU, QLinearD = self.get_linear_type()
        actU, actD = self.get_act_type()

        self.quantizer = Quantizer(
            config.hidden_size,
            config.pod_rank,
            config.input_clip_ratio,
            act_dtype=actU
        )
        self.up_proj = QLinearU.from_float(self.up_proj, pod_rank=config.pod_rank)
        self.gate_proj = QLinearU.from_float(self.gate_proj, pod_rank=config.pod_rank)
        self.down_proj = torch.nn.Sequential(
            quad.modules.OnlineHadamard(self.intermediate_size),
            Quantizer(config.hidden_size, 0, config.input_clip_ratio, act_dtype=actD),
            QLinearD.from_float(self.down_proj, extra_out=config.pod_rank),
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class QuadQwen2ForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        self._expand_embedding()
        self._expand_lm_head()
        for layer_idx, layer in tqdm(enumerate(self.model.layers), total=len(self.model.layers), desc="init model..."):
            layer.self_attn = QuadQwen2Attention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuadQwen2MLP(config=config)
        self.model.norm = quad.modules.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cache_dtype = "float16"

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
