#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import nn
import transformers
import math
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from llava.constants import IMAGE_TOKEN_INDEX
from transformers.utils import ModelOutput
  
import os 

def custom_update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    cache_name, cache = self._extract_past_from_model_output(outputs)
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if len(model_kwargs['attention_mask'].shape) == 4:
        cur_mask = model_kwargs['attention_mask']
        attention_mask_2D = torch.ones((cur_mask.shape[0], cur_mask.shape[2]), device=cur_mask.device)
        model_kwargs['attention_mask'] = attention_mask_2D
    
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
    else:
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
        ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
    return model_kwargs



def forward_custom_qwen2(
      self,
      input_ids: torch.LongTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[List[torch.FloatTensor]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      cache_position: Optional[torch.LongTensor] = None,
  ) -> Union[Tuple, BaseModelOutputWithPast]:
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states = (
          output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      )
      use_cache = use_cache if use_cache is not None else self.config.use_cache
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict


      to_adapt = None
      layer_ids = os.getenv('LAYERS')
      if layer_ids != 'None':
        to_adapt = [i for i in range(int(layer_ids.split(',')[0]), int(layer_ids.split(',')[1]) + 1)]

      if (input_ids is None) ^ (inputs_embeds is not None):
          raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

      if self.gradient_checkpointing and self.training:
          if use_cache:
              logger.warning_once(
                  "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
              )
              use_cache = False

      # kept for BC (non `Cache` `past_key_values` inputs)
      return_legacy_cache = False
      if use_cache and not isinstance(past_key_values, Cache):
          return_legacy_cache = True
          if past_key_values is None:
              past_key_values = DynamicCache()
          else:
              past_key_values = DynamicCache.from_legacy_cache(past_key_values)
              logger.warning_once(
                  "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                  "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                  "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
              )

      
      if inputs_embeds is None:
          inputs_embeds = self.embed_tokens(input_ids)

      if cache_position is None:
          past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
          cache_position = torch.arange(
              past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
          )
      
      if position_ids is None:
          position_ids = cache_position.unsqueeze(0)
      
      if input_ids is not None:
          for layer in self.layers:
              if hasattr(layer.self_attn, '_last_input_ids'):
                  layer.self_attn._last_input_ids = input_ids
     
      causal_mask = attention_mask
      if causal_mask is not None and len(causal_mask.shape) == 2:
        causal_mask = None

      hidden_states = inputs_embeds
    
      position_embeddings = self.rotary_emb(hidden_states, position_ids)

      all_hidden_states = () if output_hidden_states else None
      all_self_attns = () if output_attentions else None
      next_decoder_cache = None

      for layer_id, decoder_layer in enumerate(self.layers):
          if output_hidden_states:
              all_hidden_states += (hidden_states,)
        
          if to_adapt is not None and layer_id not in to_adapt:
              causal_mask = None
          elif causal_mask is not None:
              causal_mask = attention_mask
          
          if self.gradient_checkpointing and self.training:
              layer_outputs = self._gradient_checkpointing_func(
                  decoder_layer.__call__,
                  hidden_states,
                  causal_mask,
                  position_ids,
                  past_key_values,
                  output_attentions,
                  use_cache,
                  cache_position,
                  position_embeddings,
              )
          else:
              layer_outputs = decoder_layer(
                  hidden_states,
                  attention_mask=causal_mask,
                  position_ids=position_ids,
                  past_key_value=past_key_values,
                  output_attentions=output_attentions,
                  use_cache=use_cache,
                  cache_position=cache_position,
                  position_embeddings=position_embeddings,
              )

          hidden_states = layer_outputs[0]

          if use_cache:
              next_decoder_cache = layer_outputs[2 if output_attentions else 1]

          if output_attentions:
              all_self_attns += (layer_outputs[1],)

      hidden_states = self.norm(hidden_states)

      if output_hidden_states:
          all_hidden_states += (hidden_states,)

      next_cache = next_decoder_cache if use_cache else None
      if return_legacy_cache:
          next_cache = next_cache.to_legacy_cache()

      if not return_dict:
          return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
      return BaseModelOutputWithPast(
          last_hidden_state=hidden_states,
          past_key_values=next_cache,
          hidden_states=all_hidden_states,
          attentions=all_self_attns,
      )

def replace_qwen_attn():
  transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = forward_custom_qwen2
  transformers.generation.utils.GenerationMixin._update_model_kwargs_for_generation = custom_update_model_kwargs_for_generation



