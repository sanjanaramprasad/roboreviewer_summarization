import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartPretrainedModel, shift_tokens_right, BartDecoderLayer, BartLearnedPositionalEmbedding, BartAttention, _make_causal_mask, _expand_mask
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput,Seq2SeqSequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import copy
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN
from BartForDataToTextGeneration_encoder_combination import BartForDataToText
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
class BartDecoderLayerMulti(nn.Module):

    def __init__(self, config : BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn2 = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        #self.self_attn2_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        #self.init_weights()

        #self.encoder_attn_1 = copy.deepcopy(self.encoder_attn)
        #self.encoder_attn_2 = copy.deepcopy(self.encoder_attn)
        #self.encoder_attn_3 = copy.deepcopy(self.encoder_attn)
        #self.encoder_attn_4 = copy.deepcopy(self.encoder_attn)

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.concat_attn = BartAttention(
                embed_dim = self.embed_dim,
                num_heads = config.decoder_attention_heads,
                dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.fc_concat1 = nn.Linear(self.embed_dim * 5, (self.embed_dim * 5) * 2)
        self.fc_concat2 = nn.Linear((self.embed_dim * 5) * 2, (self.embed_dim * 5))
        self.fc_concat3 = nn.Linear(self.embed_dim * 5, self.embed_dim )
        #self.init_weights()


    def _make_duplicate_attns(self):
        self.encoder_attn_1 = copy.deepcopy(self.encoder_attn)
        self.encoder_attn_2 = copy.deepcopy(self.encoder_attn)
        self.encoder_attn_3 = copy.deepcopy(self.encoder_attn)
        self.encoder_attn_4 = copy.deepcopy(self.encoder_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states1: Optional[torch.Tensor] = None,
        encoder_attention_mask1: Optional[torch.Tensor] = None,
        encoder_hidden_states2: Optional[torch.Tensor] = None,
        encoder_attention_mask2: Optional[torch.Tensor] = None,
        encoder_hidden_states3: Optional[torch.Tensor] = None,
        encoder_attention_mask3: Optional[torch.Tensor] = None,
        encoder_hidden_states4: Optional[torch.Tensor] = None,
        encoder_attention_mask4: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
	    decoder_combination = 'addition'
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        #use_cache = False
        residual = hidden_states
        #if past_key_value:
        #print("PS Value", len(past_key_value))
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        ##print("self_attn_past_key_value", self_attn_past_key_value)
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        if encoder_hidden_states is not None:
            residual = hidden_states
        ##print("as", attention_mask)
        ##print("HS", hidden_states.shape)
        ##print("key_value_states", encoder_hidden_states.shape)
        ##print("attention_mask", encoder_attention_mask.shape)
        def cross_attn_block(encoder_attn, encoder_hidden_states, encoder_attention_mask, hidden_states, residual, cross_attn_past_key_value):
            # Cross-Attention Block
            cross_attn_present_key_value = None
            cross_attn_weights = None
            #print(encoder_hidden_states is None)
            if encoder_hidden_states is not None:
                #residual = hidden_states

                # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
                #cross_attn_past_key_value = past_key_value[beg_id:end_id] if past_key_value is not None else None
                enc_hidden_states, cross_attn_weights, cross_attn_present_key_value = encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            enc_hidden_states = F.dropout(enc_hidden_states, p=self.dropout, training=self.training)
            #hidden_states = residual + hidden_states
            #hidden_states = encoder_attn_layer_norm(hidden_states)
            #print("CROSS ATTN WEIGHTS", cross_attn_weights.shape)
            return enc_hidden_states, cross_attn_present_key_value

        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
        hidden_states_0, cross_attn_present_key_value_0 = cross_attn_block(self.encoder_attn, encoder_hidden_states, encoder_attention_mask, hidden_states, residual, cross_attn_past_key_value)

        cross_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
        hidden_states_1, cross_attn_present_key_value_1 = cross_attn_block(self.encoder_attn_1, encoder_hidden_states1, encoder_attention_mask1, hidden_states, residual, cross_attn_past_key_value)

        cross_attn_past_key_value = past_key_value[6:8] if past_key_value is not None else None
        hidden_states_2, cross_attn_present_key_value_2 = cross_attn_block(self.encoder_attn_2, encoder_hidden_states2, encoder_attention_mask2, hidden_states, residual, cross_attn_past_key_value)
        
        cross_attn_past_key_value = past_key_value[8:10] if past_key_value is not None else None
        hidden_states_3, cross_attn_present_key_value_3 = cross_attn_block(self.encoder_attn_3, encoder_hidden_states3, encoder_attention_mask3, hidden_states, residual, cross_attn_past_key_value)
        

        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states_4, cross_attn_present_key_value_4 = cross_attn_block(self.encoder_attn_4, encoder_hidden_states4, encoder_attention_mask4, hidden_states, residual, cross_attn_past_key_value)
        # add cross-attn to positions 3,4 of present_key_value tuple
        if decoder_combination == 'addition':
            hidden_states_0 = hidden_states_0 * torch.sigmoid(self.w0(hidden_states_0))
            hidden_states_1 = hidden_states_1 * torch.sigmoid(self.w1(hidden_states_1))
            hidden_states_2 = hidden_states_2 * torch.sigmoid(self.w2(hidden_states_2))
            hidden_states_2 = hidden_states_2 * torch.sigmoid(self.w2(hidden_states_2))
            hidden_states_2 = hidden_states_2 * torch.sigmoid(self.w2(hidden_states_2))

            hidden_states_all = hidden_states_0 + hidden_states_1 + hidden_states_2 + hidden_states_3 + hidden_states_4
            '''hidden_states_all = hidden_states_0 + hidden_states_1 + hidden_states_2 + hidden_states_3 + hidden_states_4
               self_attn2_past_key_value = past_key_value[10:12] if past_key_value is not None else None
               hidden_states_all, self_attn2_weights, self_attn2_present_key_value = self.self_attn2(
               hidden_states = hidden_states_all,
                past_key_value = self_attn2_past_key_value,
                attention_mask = None,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
               ) '''             




        else:
                hidden_states_concat = torch.cat([hidden_states_0, hidden_states_1, hidden_states_2, hidden_states_3, hidden_states_4], dim =2)
                ## concatenate hidden states 
                hidden_states_concat = self.activation_fn(self.fc_concat1(hidden_states_concat))
                hidden_states_concat = F.dropout(hidden_states_concat, p=self.activation_dropout, training=self.training)
                hidden_states_concat = self.activation_fn(self.fc_concat2(hidden_states_concat))
                hidden_states_concat = F.dropout(hidden_states_concat, p=self.activation_dropout, training=self.training)
                hidden_states_concat = self.fc_concat3(hidden_states_concat)
                hidden_states_concat = F.dropout(hidden_states_concat, p=self.dropout, training=self.training)
                
                concat_attn_past_key_value = past_key_value[10:12] if past_key_value is not None else None
                hidden_states_all, concat_attn_weights, concat_attn_present_key_value = self.concat_attn(
                hidden_states = hidden_states_concat,
                past_key_value = concat_attn_past_key_value,
                attention_mask = None,
                layer_head_mask=layer_head_mask,
            	output_attentions=output_attentions,
        	)

                
        
        ## self attention over concatenated hidden states 
        ## pass the hidden states through a fcn 

        hidden_states = hidden_states_all + residual
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        present_key_value = present_key_value + cross_attn_present_key_value_0 + cross_attn_present_key_value_1 + cross_attn_present_key_value_2 + cross_attn_present_key_value_3 + cross_attn_present_key_value_4 + self_attn2_present_key_value
        if decoder_combination!= 'addition':
	        present_key_value = present_key_value + concat_attn_present_key_value
	

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_present_key_value_0)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    

class BartDecoderMulti(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayerMulti(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states1=None,
        encoder_attention_mask1=None,
        encoder_hidden_states2=None,
        encoder_attention_mask2=None,
        encoder_hidden_states3=None,
        encoder_attention_mask3=None,
        encoder_hidden_states4=None,
        encoder_attention_mask4=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_forward_strategy = 'forward',
        decoder_combination = 'addition',
        return_dict=None,
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
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.
                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_hidden_states1 is not None and encoder_attention_mask1 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask1 = _expand_mask(encoder_attention_mask1, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_hidden_states2 is not None and encoder_attention_mask2 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask2 = _expand_mask(encoder_attention_mask2, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_hidden_states3 is not None and encoder_attention_mask3 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask3 = _expand_mask(encoder_attention_mask3, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_hidden_states4 is not None and encoder_attention_mask4 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask4 = _expand_mask(encoder_attention_mask4, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            

            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states1=encoder_hidden_states1,
                    encoder_attention_mask1=encoder_attention_mask1,
                    encoder_hidden_states2=encoder_hidden_states2,
                    encoder_attention_mask2=encoder_attention_mask2,
                    encoder_hidden_states3=encoder_hidden_states3,
                    encoder_attention_mask3=encoder_attention_mask3,
                    encoder_hidden_states4=encoder_hidden_states4,
                    encoder_attention_mask4=encoder_attention_mask4,

                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    decoder_combination = decoder_combination,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartEncoderShared():
    def __init__(self, enc, layers ):
        ind = 0
        own_layers = enc.layers[3:]
        for shared_layer in layers:
            own_layers.insert(ind, shared_layer)
            ind +=1
        enc.layers = own_layers

class BartForDataToTextDecoderMod(BartForDataToText):
    base_model_prefix = "model" 
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    
    def __init__(self, config: BartConfig):
        super().__init__(config)
        ##print(config.encoder_layers)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        self.encoder = BartEncoder(config, self.shared)
        

        #config_decoder = copy.deepcopy(config)
        #config_decoder.d_model = 1200
        self.decoder = BartDecoderMulti(config,self.shared)
        self.decoder.layers = nn.ModuleList([BartDecoderLayerMulti(config) for _ in range(config.decoder_layers)])

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        print("DIM", config.d_model)
        self.init_weights()

        
        
    def _make_duplicate_decoder_layer_attns(self):
        for each_layer in self.decoder.layers:
            each_layer._make_duplicate_attns()
            
    def _make_duplicate_encoders(self, layer_share = True):
        self.encoder1 = copy.deepcopy(self.encoder)
        self.encoder2 = copy.deepcopy(self.encoder)
        self.encoder3 = copy.deepcopy(self.encoder)
        self.encoder4 = copy.deepcopy(self.encoder)
        if layer_share:
            BartEncoderShared(self.encoder1, self.encoder.layers[:3])
            BartEncoderShared(self.encoder2, self.encoder.layers[:3])
            BartEncoderShared(self.encoder3, self.encoder.layers[:3])
            BartEncoderShared(self.encoder4, self.encoder.layers[:3])
        
    
    def forward(
        self,
        input_ids_col0 = None,
        input_ids_col1 = None,
        input_ids_col2 = None, 
        input_ids_col3 = None,
        input_ids_col4 = None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        attention_mask_col4 = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_col0 = None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_combination = 'addition'
    ):
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs_list = []
        attn_mask_list = [attention_mask_col0, attention_mask_col1, attention_mask_col2, \
                            attention_mask_col3, attention_mask_col4]

        if not (input_ids_col0 is None):
            encoder_outputs_col0 = self._get_encoder_outputs(
                        encoder = self.encoder, 
                        encoder_outputs = encoder_outputs_col0, 
                        input_ids = input_ids_col0,
                        attention_mask = attention_mask_col0,
                        head_mask = head_mask,
                        inputs_embeds = inputs_embeds,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        return_dict = return_dict)

        if not (input_ids_col1 is None):
            encoder_outputs_col1 = self._get_encoder_outputs(
                        encoder = self.encoder1, 
                        encoder_outputs = encoder_outputs_col1, 
                        input_ids = input_ids_col1,
                        attention_mask = attention_mask_col1,
                        head_mask = head_mask,
                        inputs_embeds = inputs_embeds,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        return_dict = return_dict)
            

        if not (input_ids_col2 is None):
            encoder_outputs_col2 = self._get_encoder_outputs(
                        encoder = self.encoder2, 
                        encoder_outputs = encoder_outputs_col2, 
                        input_ids = input_ids_col2,
                        attention_mask = attention_mask_col2,
                        head_mask = head_mask,
                        inputs_embeds = inputs_embeds,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        return_dict = return_dict)
            
        
        if not (input_ids_col3 is None):
            encoder_outputs_col3 = self._get_encoder_outputs(
                        encoder = self.encoder3, 
                        encoder_outputs = encoder_outputs_col3, 
                        input_ids = input_ids_col3,
                        attention_mask = attention_mask_col3,
                        head_mask = head_mask,
                        inputs_embeds = inputs_embeds,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        return_dict = return_dict)
            
        
        if not (input_ids_col4 is None):
            encoder_outputs_col4 = self._get_encoder_outputs(
                        encoder = self.encoder4, 
                        encoder_outputs = encoder_outputs_col4, 
                        input_ids = input_ids_col4,
                        attention_mask = attention_mask_col4,
                        head_mask = head_mask,
                        inputs_embeds = inputs_embeds,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        return_dict = return_dict)
            
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                all_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            
            encoder_hidden_states=encoder_outputs_col0[0],
            encoder_attention_mask=attention_mask_col0,
            
            encoder_hidden_states1=encoder_outputs_col1[0],
            encoder_attention_mask1=attention_mask_col1,
            
            encoder_hidden_states2=encoder_outputs_col2[0],
            encoder_attention_mask2=attention_mask_col2,
            
            encoder_hidden_states3=encoder_outputs_col3[0],
            encoder_attention_mask3=attention_mask_col3,
            
            encoder_hidden_states4=encoder_outputs_col4[0],
            encoder_attention_mask4=attention_mask_col4,
            
            head_mask=decoder_head_mask,
            cross_attn_head_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            decoder_combination = decoder_combination,
            return_dict=return_dict,
        )
        
        if not return_dict:
            outputs =  decoder_outputs + encoder_outputs
            
        else:
            outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
            )
            
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            #print("Adding LM logits")
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
       )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        attention_mask_col4 = None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_col0 =None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        decoder_combination = 'addition',
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids_col0": None,
            "input_ids_col1": None,
            "input_ids_col2": None,
            "input_ids_col3": None,
            "input_ids_col4": None,
            "encoder_outputs_col0": encoder_outputs_col0,
            "encoder_outputs_col1": encoder_outputs_col1,
            "encoder_outputs_col2": encoder_outputs_col2,
            "encoder_outputs_col3": encoder_outputs_col3,
            "encoder_outputs_col4": encoder_outputs_col4,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask_col0": attention_mask_col0,
            "attention_mask_col1": attention_mask_col1,
            "attention_mask_col2": attention_mask_col2,
            "attention_mask_col3": attention_mask_col3,
            "attention_mask_col4": attention_mask_col4,
            "decoder_combination": decoder_combination,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)

        }
