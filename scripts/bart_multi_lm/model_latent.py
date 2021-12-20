import torch
from dataclasses import dataclass
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartModel, BartPretrainedModel, shift_tokens_right, BartDecoder, BartLearnedPositionalEmbedding,  _make_causal_mask, _expand_mask
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput,Seq2SeqSequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import copy
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN
import random
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

@dataclass
class BARTSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_logits_individual : Optional[Tuple[Tuple[torch.FloatTensor]]] = None

class Mixture(nn.Module):
    def __init__(self, num_inputs):
        super(Mixture, self).__init__()
        self.num_inputs = num_inputs
        self.softmax_gate = nn.Softmax(dim = 0)
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(3)) for i in range(0, self.num_inputs)])
        
        
    def forward(self, v0, v1, v2, t=None):
        #softmaxed_weights = torch.nn.functional.softmax(self.weights)
        if not t :
            idx = 0
        
        if t:
            idx = t 

        v_mixt = []
        for n in range(0, v0.shape[0]):
            if self.num_inputs == v0.shape[0] :
                idx = n
            W = self.weights[idx]
            W = self.softmax_gate(W)
            print('v0', torch.stack([v0[n].unsqueeze(0), v1[n].unsqueeze(0), v2[n].unsqueeze(0) ], dim = 0).shape)
            v_t = (W[0] * v0[n][:, None]) + (W[1] * v1[n][:, None]) + (W[2] * v2[n][:, None])
            v_mixt.append(v_t.t())
        #print(torch.cat(v_mixt).shape, v0.shape)
        #print(W[0], W[1], W[2])
        return torch.cat(v_mixt)


class BartEncoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.k_proj_pop = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_int = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_ptext = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.v_proj_pop = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj_int = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj_ptext = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _attr_key(self, hidden_states, bsz, attribute_key):
        if attribute_key == 'pop':
            return self._shape(self.k_proj_pop(hidden_states), -1, bsz)
        elif attribute_key == 'int':
            return self._shape(self.k_proj_int(hidden_states), -1, bsz)
        if attribute_key == 'out':
            return self._shape(self.k_proj_out(hidden_states), -1, bsz)
        if attribute_key == 'ptext':
            return self._shape(self.k_proj_ptext(hidden_states), -1, bsz)

    def _attr_value(self, hidden_states, bsz, attribute_key):
        if attribute_key == 'pop':
            return self._shape(self.v_proj_pop(hidden_states), -1, bsz)
        elif attribute_key == 'int':
            return self._shape(self.v_proj_int(hidden_states), -1, bsz)
        if attribute_key == 'out':
            return self._shape(self.v_proj_out(hidden_states), -1, bsz)
        if attribute_key == 'ptext':
            return self._shape(self.v_proj_ptext(hidden_states), -1, bsz)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attribute_key = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            print("ATTRIBUTE KEY", attribute_key)
            key_attr_states = self._attr_key(hidden_states=hidden_states, bsz=bsz,attribute_key=attribute_key)
            value_attr_states = self._attr_value(hidden_states=hidden_states, bsz=bsz,attribute_key=attribute_key)
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = key_states.add(key_attr_states)
            value_states = value_states.add(value_attr_states)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        attribute_key = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            attribute_key = attribute_key,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        attribute_key = None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        attribute_key,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        attribute_key=attribute_key,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        attribute_key = None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                attribute_key=attribute_key,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
class BartForDataToTextGeneration_MultiLM(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        #self.model1 = BartModel(config)
        #self.model2 = BartModel(config)
        self.register_buffer("final_logits_bias0", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias1", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias2", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias3", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias4", torch.zeros((1, self.model.shared.num_embeddings)))

        self.softmax_logits = nn.LogSoftmax(dim = 2)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.activation_fn = ACT2FN['relu']
        self.LayerNorm = nn.LayerNorm(config.d_model)
        '''self.weight_vect0 = nn.Linear(config.d_model, 1, bias = False)
        self.weight_vect1 = nn.Linear(config.d_model, 1, bias = False)
        self.weight_vect2 = nn.Linear(config.d_model, 1, bias = False)
        self.weight_vect3 = nn.Linear(config.d_model, 1, bias = False)'''
        #self.lm_head1 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        #self.lm_head2 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        #self.lm_head3 = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        #self.lm_combine = Mixture(num_inputs=1)
        self.weigh_context = nn.Linear(config.d_model * 4 , config.d_model * 2)
        self.weigh_context1 = nn.Linear(config.d_model*2 , config.d_model)
        self.weigh_context_final = nn.Linear(config.d_model, 4)
        ##self.weigh_context3 = nn.Linear(config.d_model , 1)
        
        self.soft_weigh = nn.Softmax(dim =2)
        self.init_weights()

    def _make_multiple_lm_heads(self):
        self.lm_head1 = copy.deepcopy(self.lm_head)
        self.lm_head2 = copy.deepcopy(self.lm_head)
        self.lm_head3 = copy.deepcopy(self.lm_head)
        #self.lm_head4 = copy.deepcopy(self.lm_head)
        return

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_func(self, final_logits_bias, new_num_tokens, old_num_tokens):
        if new_num_tokens <= old_num_tokens:
            new_bias = final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=final_logits_bias.device)
            new_bias = torch.cat([final_logits_bias, extra_bias], dim=1)
        return new_bias


    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias0.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias0, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias0", new_bias)

        old_num_tokens = self.final_logits_bias1.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias1, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias1", new_bias)

        old_num_tokens = self.final_logits_bias2.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias2, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias2", new_bias)

        old_num_tokens = self.final_logits_bias3.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias3, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias3", new_bias)

        '''old_num_tokens = self.final_logits_bias4.shape[-1]
        new_bias = self._resize_func(self.final_logits_bias4, new_num_tokens, old_num_tokens)
        self.register_buffer("final_logits_bias4", new_bias)'''

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.lm_head1 = new_embeddings
        self.lm_head2 = new_embeddings
        self.lm_head3 = new_embeddings
        #self.lm_head4 = new_embeddings

    def _get_sentence_vectors(self, encoder_output_list, bos_id_list):
        vector_list = []
        vector_attention = []
        max_len = encoder_output_list[0][0].shape[0]
        embed_dim = encoder_output_list[0][0].shape[1]
        batch_size = encoder_output_list[0].shape[0]
        '''print("MAX LEN", max_len)
        print("EMB DIM", embed_dim)
        print('BATCH SIZE', batch_size)'''
        #print('ENC OUT LIST', len(encoder_output_list), encoder_output_list[0].shape)
        #print('BOS LIST', len(bos_id_list), bos_id_list[0].shape)

        #print(encoder_output_list[0], bos_id_list[0])
        #print(list(zip(encoder_output_list, bos_id_list))[0])

        for batch_id in range(0, batch_size):
            batch_vector_list = []

            for enc_last_hidden_state, bos_ids in list(zip(encoder_output_list, bos_id_list)):
                enc_last_hs_vectors = enc_last_hidden_state[batch_id]
                sentence_output = []

                for i in bos_ids[batch_id].tolist():
                    if i != -2:
                        sentence_output.append(enc_last_hs_vectors[i].tolist())

                batch_vector_list += sentence_output
            vector_list.append(batch_vector_list)

        vector_list_padded= []
        vector_attentions = []

        for vect_list in vector_list:
            vector_attention = [1] * len(vect_list) + [0] * (max_len - len(vect_list))
            vect_list += [[0] * embed_dim] * (max_len - len(vect_list))

            vector_list_padded.append(vect_list)
            vector_attentions.append(vector_attention)

        vector_list = torch.as_tensor(vector_list_padded, device = encoder_output_list[0][0].device)
        vector_attentions = torch.as_tensor(vector_attentions, device = encoder_output_list[0][0].device)
        return vector_list, vector_attentions

    def forward(
        self,
        input_ids_col0 = None,
        input_ids_col1 = None,
        input_ids_col2 = None, 
        input_ids_col3 = None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_col0 = None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        decoder_time_step = None,
        cross_attn_head_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        outputs0 = self.model(
            input_ids_col0,
            attention_mask=attention_mask_col0,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs_col0,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[0] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attribute_key = 'pop',
            return_dict=return_dict,
        )

        outputs1 = self.model(
            input_ids_col1,
            attention_mask=attention_mask_col1,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs_col1,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[1] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attribute_key = 'int',
            return_dict=return_dict,
        )

        outputs2 = self.model(
            input_ids_col2,
            attention_mask=attention_mask_col2,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs_col2,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[2] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attribute_key = 'out',
            return_dict=return_dict,
        )

        outputs3 = self.model(
            input_ids_col3,
            attention_mask=attention_mask_col3,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs_col3,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values[3] if past_key_values else None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attribute_key = 'ptext',
            return_dict=return_dict,
        )

       

        
        #alphas_0 = self.weigh_context(outputs0[0])
        #alphas_1 = self.weigh_context(outputs1[0])
        #alphas_2 = self.weigh_context(outputs2[0])
        #alphas_3 = self.weigh_context(outputs3[0])
        context_vect = torch.cat([outputs0[0], outputs1[0], outputs2[0], outputs3[0]], dim = -1)
        #context_vect = torch.max(context_vect, dim = 0)[0]
        context_vect = self.activation_fn(self.weigh_context(context_vect)) ## D * 2
        context_vect = self.activation_fn(self.weigh_context1(context_vect)) ## D 
        alphas = self.weigh_context_final(context_vect) ## 4

        #alphas = torch.cat([alphas_0, alphas_1, alphas_2, alphas_3], dim = -1)
        alphas = self.soft_weigh(alphas) 
        #print('APLHAS', alphas)
        #alphas = self.soft_weigh(alphas)
        alphas_ind = torch.argmax(alphas, 2, keepdim=True)
        one_hot = torch.FloatTensor(alphas.shape)
        alphas_ind = alphas_ind.to(device = one_hot.device)
        one_hot.zero_()
        one_hot.scatter_(2, alphas_ind, 1)
        alphas = one_hot
        alphas = alphas.to(device = outputs3[0].device)
        #print('ONE HOT', one_hot)
        #print("ALPHAS", alphas.shape, alphas[0][:, 0][:, None])
        #alphas = alphas[0]'''
        #print('WEIGHTS', alphas)
        #print(input_ids)
        lm_logits0 = self.lm_head(outputs0[0]) + self.final_logits_bias0
        lm_logits1 = self.lm_head1(outputs1[0]) + self.final_logits_bias1
        lm_logits2 = self.lm_head2(outputs2[0]) + self.final_logits_bias2
        lm_logits3 = self.lm_head3(outputs3[0]) + self.final_logits_bias3
        #lm_logits4 = self.lm_head4(outputs4[0]) + self.final_logits_bias4
        lm_logits0 = self.softmax_logits(lm_logits0)
        lm_logits1 = self.softmax_logits(lm_logits1)
        lm_logits2 = self.softmax_logits(lm_logits2)
        lm_logits3 = self.softmax_logits(lm_logits3)
        #lm_logits4 = self.softmax_logits(lm_logits4)'''

        #print("LOGITS SINGLE", lm_logits0.shape)
        lm_logits = [ alphas[batch_id][:, 0][:, None] *  lm_logits0[batch_id].unsqueeze(0) + alphas[batch_id][:, 1][:, None] *  lm_logits1[batch_id].unsqueeze(0)  + \
            alphas[batch_id][:, 2][:, None] *  lm_logits2[batch_id].unsqueeze(0) \
                + alphas[batch_id][:, 3][:, None] *  lm_logits3[batch_id].unsqueeze(0) \
                for batch_id in range(0, lm_logits0.shape[0])]
        #lm_logits = torch.cat([lm_logits1[batch_id].unsqueeze(0) for batch_id in range(0, lm_logits0.shape[0])])
        #print('lm combined', lm_logits.shape)
        lm_logits = torch.cat(lm_logits)
        #lm_logits = self.softmax_logits(lm_logits)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits) + outputs3[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        #print(lm_logits0[0].unsqueeze(0).shape)        
        lm_logits_list = [torch.stack([alphas[batch_id][:,0]  , alphas[batch_id][:,1] \
                  , alphas[batch_id][:,2] , alphas[batch_id][:,3]]) \
                for batch_id in range(0, lm_logits0.shape[0])]
        lm_logits_list = torch.stack(lm_logits_list)
        #print(lm_logits_list.shape)
        return BARTSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=[outputs0.past_key_values, outputs1.past_key_values, outputs2.past_key_values, outputs3.past_key_values],
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
            lm_logits_individual = lm_logits_list
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_col0 =None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        **kwargs
    ):
        decoder_time_step =  decoder_input_ids.shape[1] - 1
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids_col0": None,
            "input_ids_col1": None,
            "input_ids_col2": None,
            "input_ids_col3": None,
            "decoder_time_step":decoder_time_step,
            "encoder_outputs_col0": encoder_outputs_col0,
            "encoder_outputs_col1": encoder_outputs_col1,
            "encoder_outputs_col2": encoder_outputs_col2,
            "encoder_outputs_col3": encoder_outputs_col3,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask_col0": attention_mask_col0,
            "attention_mask_col1": attention_mask_col1,
            "attention_mask_col2": attention_mask_col2,
            "attention_mask_col3": attention_mask_col3,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)

     }


    @staticmethod
    def _reorder_cache(past, beam_idx):
        past_all = []
        for past_idx in past:
            reordered_past = ()
            for layer_past in past_idx:
                reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
            past_all.append(reordered_past)
        return past_all
