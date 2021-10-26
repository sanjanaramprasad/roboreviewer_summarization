import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartModel, BartPretrainedModel, shift_tokens_right, BartDecoderLayer, BartLearnedPositionalEmbedding,  _make_causal_mask, _expand_mask
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
        self.softmax_logits = nn.LogSoftmax(dim = 2)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        #self.lm_head1 = nn.Linear(config.d_model, self.model1.shared.num_embeddings, bias=False)
        #self.lm_head2 = nn.Linear(config.d_model, self.model2.shared.num_embeddings, bias=False)
        #self.lm_combine = Mixture(num_inputs=1)
        self.weigh_context = nn.Linear(config.d_model * 3, 3)
        self.init_weights()

    def _make_multiple_lm_heads(self):
        self.lm_head1 = copy.deepcopy(self.lm_head)
        self.lm_head2 = copy.deepcopy(self.lm_head)
        self.lm_head3 = copy.deepcopy(self.lm_head)
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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.lm_head1 = new_embeddings
        self.lm_head2 = new_embeddings
        self.lm_head3 = new_embeddings


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
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

       
        #print(outputs0[0].shape) 
        #print(torch.cat([outputs0[0], outputs1[0], outputs2[0]], dim = -1).shape) 
        alphas = self.weigh_context(torch.cat([outputs0[0], outputs1[0], outputs2[0], outputs3[0]], dim = -1))
        alphas = alphas[0]
        #print('WEIGHTS', alphas.shape)
        #print(input_ids)
        lm_logits0 = self.lm_head(outputs0[0]) + self.final_logits_bias0
        lm_logits1 = self.lm_head1(outputs1[0]) + self.final_logits_bias1
        lm_logits2 = self.lm_head2(outputs2[0]) + self.final_logits_bias2
        lm_logits3 = self.lm_head2(outputs3[0]) + self.final_logits_bias3
        lm_logits0 = self.softmax_logits(lm_logits0)
        lm_logits1 = self.softmax_logits(lm_logits1)
        lm_logits2 = self.softmax_logits(lm_logits2)
        lm_logits3 = self.softmax_logits(lm_logits3)

        lm_logits = [ alphas[batch_id][0] *  lm_logits0 + alphas[batch_id][1] *  lm_logits1  + alphas[batch_id][2] *  lm_logits2 + alphas[batch_id][3] *  lm_logits3 \
                for batch_id in range(0, lm_logits0.shape[0])]
        lm_logits = torch.cat(lm_logits)
        #print('lm combined', lm_logits.shape)
        #lm_logits = self.softmax_logits(lm_logits)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits0.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits0,) + outputs0[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=[outputs0.past_key_values, outputs1.past_key_values, outputs2.past_key_values],
            decoder_hidden_states=outputs0.decoder_hidden_states,
            decoder_attentions=outputs0.decoder_attentions,
            cross_attentions=outputs0.cross_attentions,
            encoder_last_hidden_state=outputs0.encoder_last_hidden_state,
            encoder_hidden_states=outputs0.encoder_hidden_states,
            encoder_attentions=outputs0.encoder_attentions,
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
