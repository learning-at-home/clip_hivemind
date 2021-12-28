# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GPT modules that do not hog your GPU memory """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from lib.models.transformer import LeanTransformer, LeanTransformerConfig
from lib.modules.sequence import GradientCheckpointingMixin

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanGPTConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"


class LeanGPTConfig(LeanTransformerConfig):
    def __init__(
        self,
        *args,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs
    ):
        super().__init__(
            *args,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            **kwargs
        )
        self.type_vocab_size = type_vocab_size


class LeanGPTEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. These embeddigns double as logits.
    """

    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)

        self.token_type_embeddings = config.get_token_type_embeddings()
        self.position_embeddings = config.get_input_position_embeddings()

        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.embedding_size != config.hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

        if self.position_embeddings is not None:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if hasattr(self, "embedding_hidden_mapping_in"):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class TiedMLMHead(nn.Module):
    def __init__(self, config, embeddings: LeanGPTEmbeddings):
        super().__init__()
        self.embeddings = embeddings

        if config.embedding_size != config.hidden_size:
            self.hidden_bias = nn.Parameter(torch.zeros(config.embedding_size))

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.activation = ACT2FN[config.hidden_act]
        self.logits_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        if hasattr(self, "hidden_bias"):
            weight = self.embeddings.embedding_hidden_mapping_in.weight.t()
            hidden_states = F.linear(input=hidden_states, weight=weight, bias=self.hidden_bias)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        weight = self.embeddings.word_embeddings.weight.t()
        logits = F.linear(input=hidden_states, weight=weight, bias=self.logits_bias)
        return logits


class LeanGPTForPreTraining(GradientCheckpointingMixin, PreTrainedModel):
    config_class = LeanGPTConfig
    base_model_prefix = "lean_albert"

    def __init__(self, config: config_class):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = LeanGPTEmbeddings(config)
        self.transformer = LeanTransformer(config)
        self.lm_head = TiedMLMHead(config, self.embeddings)
        self.init_weights()

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def _init_weights(self, module):
        """Initialize the weights."""
        logger.warning("INIT IS MESSED UP, GO UNMESS IT!")
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            ignored_labels = torch.full_like(labels[..., :1], fill_value=-100)
            shift_labels = torch.cat([labels[..., 1:], ignored_labels], dim=1)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), shift_labels.view(-1),
                                   reduction='mean', ignore_index=-100)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )