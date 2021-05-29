import torch
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartPretrainedModel, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput,Seq2SeqSequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import copy

class BartForDataToText(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    
    def __init__(self, config: BartConfig):
        super().__init__(config)
        #print(config.encoder_layers)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        
        self.encoder = BartEncoder(config, self.shared)
        

        config_decoder = copy.deepcopy(config)
        config_decoder.d_model = 1200
        self.shared_decoder = nn.Embedding(config_decoder.vocab_size, config_decoder.d_model, padding_idx)
        #self.decoder = BartDecoder(config, self.shared)

        self.decoder1 = BartDecoder(config_decoder,self.shared_decoder)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared_decoder.num_embeddings)))
        #self.lm_head = nn.Linear(config.d_model , self.shared.num_embeddings, bias=False)
        self.lm_head = nn.Linear(config_decoder.d_model, self.shared_decoder.num_embeddings, bias=False)
        self.fc0 = nn.Linear(config.d_model, 240)
        self.fc1 = nn.Linear(config.d_model, 240)
        self.fc2 = nn.Linear(config.d_model, 240)
        self.fc3 = nn.Linear(config.d_model, 240)
        self.fc4 = nn.Linear(config.d_model, 240)
        print("DIM", config.d_model)
        self.init_weights()
        

        
    def _make_duplicate_encoders(self):
        self.encoder1 = copy.deepcopy(self.encoder)
        self.encoder2 = copy.deepcopy(self.encoder)
        self.encoder3 = copy.deepcopy(self.encoder)
        self.encoder4 = copy.deepcopy(self.encoder)
        
    def get_input_embeddings(self):
        return self.shared 
    
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder = self.shared
        self.encoder1 = self.shared
        self.encoder2 = self.shared
        self.encoder3 = self.shared
        self.encoder4 = self.shared
        
    def get_encoders(self):
        return self.encoder, self.encoder1, \
            self.encoder2, self.encoder3, self.encoder4
    
    def get_decoder(self):
        self.decoder1
        
    def get_output_embeddngs(self):
        #eturn self.lm_head
        return None

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        print("RESIZING", new_embeddings.shape)
        return new_embeddings
    
    def _get_encoder_outputs(self, 
            encoder , 
            encoder_outputs, 
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict):
        
        if encoder_outputs is None:
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        return encoder_outputs
        
    
    def _get_sum_encoder_outputs(self,
            encoder_output_list):
        encoder_outputs = []
        for i in range(0,3):
            if len(encoder_output_list[0]) > i:
                added_enc_outputs_i = torch.cat(tuple([enc[i] for enc in encoder_output_list]), dim = 0)

                added_enc_outputs_i = torch.sum(added_enc_outputs_i, dim = 0)
                added_enc_outputs_i = added_enc_outputs_i.unsqueeze(0)
                encoder_outputs.append(added_enc_outputs_i)

        added_enc_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        #print(added_enc_outputs)
        return added_enc_outputs


    def _get_concat_encoder_outputs(self, 
        encoder_outputs_list):

        encoder_outputs = []
        for i in range(0,3):
            if len(encoder_outputs_list[0]) > i: 
                added_enc_outputs_i = torch.cat([enc[i] for enc in encoder_outputs_list],2)
                encoder_outputs.append(added_enc_outputs_i)
            
        added_enc_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        #print(added_enc_outputs)
        return added_enc_outputs
        
    
    def _get_attention_masks_OR(self, 
        attention_mask_list ):

            #all_attn_outputs = torch.cat(attention_mask_list, 1)

            all_attn_outputs = torch.cat(tuple(attention_mask_list), 0)
            added_enc_attns = torch.Tensor.float(all_attn_outputs).mean(0).tolist()
            added_enc_attns = [1 if each > 0.5 else 0 for each in added_enc_attns]
            #added_enc_attns = torch.as_tensor([added_enc_attns])
            added_enc_attns = torch.as_tensor([added_enc_attns] , device = attention_mask_list[0].device)
            return added_enc_attns
    
    def _forward_pass(self, encoder_outputs, fcn):
        enc_outputs = []
        for i in range(0,3):
            if len(encoder_outputs) > i:
                enc_outputs.append(fcn(encoder_outputs[i]))

        encoder_outputs = BaseModelOutput(
                last_hidden_state=enc_outputs[0],
                hidden_states=enc_outputs[1] if len(enc_outputs) > 1 else None,
                attentions=enc_outputs[2] if len(enc_outputs) > 2 else None,
            )
        return encoder_outputs

        
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
        encoder_combination_type = 'addition'
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
        print(attention_mask_col0.shape)
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
            #encoder_outputs_list.append(encoder_outputs_col0)

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
            #encoder_outputs_list.append(encoder_outputs_col1)

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
            #encoder_outputs_list.append(encoder_outputs_col2)
        
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
            #encoder_outputs_list.append(encoder_outputs_col3)
        
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
            #encoder_outputs_list.append(encoder_outputs_col4)
        
        ## Since BART decoder gets the same input as the encoder shifted right
        ## concatenate the source input_ids fed to different encoders to feed to BART decoder
        '''all_input_ids = torch.cat((
                input_ids_punchline_texts,
                input_ids_punchline_effects,
                input_ids_populations,
                input_ids_interventions,
                input_ids_outcomes),0)'''

        if encoder_combination_type == 'linearize':
            encoder_outputs_col0 = self._forward_pass(encoder_outputs_col0, self.fc0)
            encoder_outputs_col1 = self._forward_pass(encoder_outputs_col1, self.fc1)
            encoder_outputs_col2 = self._forward_pass(encoder_outputs_col2, self.fc2)
            encoder_outputs_col3 = self._forward_pass(encoder_outputs_col3, self.fc3)
            encoder_outputs_col4 = self._forward_pass(encoder_outputs_col4, self.fc4)


        print("ENCODER SHAPES", encoder_outputs_col4[0].shape)
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                all_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            
        encoder_outputs = [encoder_outputs_col0, encoder_outputs_col1, encoder_outputs_col2, \
                            encoder_outputs_col3, encoder_outputs_col4]

        encoder_outputs_list = [each for each in encoder_outputs if not (each is None)]

        if encoder_combination_type =='addition':
        
            encoder_outputs = self._get_sum_encoder_outputs(
                    encoder_outputs_list
                )
            print("POST SUMMED SHAPE", encoder_outputs[0].shape)
        

            if attention_mask_col0 is None:
                attn_mask = attention_mask_col0
            else:
                attn_mask = self._get_attention_masks_OR(
                    [attn_mask for attn_mask in attn_mask_list if not (attn_mask is None)]

                )

        elif encoder_combination_type == 'linearize':
            encoder_outputs = self._get_concat_encoder_outputs([encoder_outputs_col0, encoder_outputs_col1, encoder_outputs_col2, \
                            encoder_outputs_col3, encoder_outputs_col4])
            #print("ENC OUTPUT", encoder_outputs.shape)
            if attention_mask_col0 is None:
                attn_mask = attention_mask_col0
            else:
                attn_mask= self._get_attention_masks_OR(
                    [attn_mask for attn_mask in attn_mask_list if not (attn_mask is None)]

                )
        print(attn_mask.shape)

        #print("ENC ATTNS", added_enc_attns)
        #print(attention_mask_punchline_texts)
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder1(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attn_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )

        #output_dummy = torch.ones(1,512,1200)
        #oz = self.dummy(output_dummy)
        #print(oz.shape)
        lm_head_output = self.lm_head(outputs[0])
        lm_logits = lm_head_output + self.final_logits_bias
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
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
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past        
    
        



