import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bart.configuration_bart import BartConfig
import torch
import torch.distributed as dist
from torch.nn import functional as F
#from BartForDataToTextGeneration_layer_sharing import BartForDataToText
from BartForDataToTextGeneration_encoder_combination import BartForDataToText
from transformers.generation_utils import GenerationMixin
##from run_experiment_linearize import LitModel
from run_experiment_encoder_combination import LitModel
from transformers import BartTokenizer
from Data2TextProcessor_loop import SummaryDataModule
import argparse
#import pandas as pd
from rouge import Rouge
from rouge_score import rouge_scorer
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
import pandas as pd
import nltk
from nltk.translate import meteor_score

class Data2TextGenerator(GenerationMixin):

    def __init__(self, model, tokenizer):
        self.model = model.model 
        self.tokenizer = tokenizer 
        self.config = self.model.config
        self.device = self.model.device
        #print(self.config.max_length)


    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask_col0: torch.LongTensor = None,
        attention_mask_col1: torch.LongTensor = None,
        attention_mask_col2: torch.LongTensor = None,
        attention_mask_col3: torch.LongTensor = None,
        attention_mask_col4: torch.LongTensor = None,
        encoder_outputs_col0: ModelOutput = None,
        encoder_outputs_col1: ModelOutput = None,
        encoder_outputs_col2: ModelOutput = None,
        encoder_outputs_col3: ModelOutput = None,
        encoder_outputs_col4: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask_col0 is not None:
            model_kwargs["attention_mask_col0"] = attention_mask_col0.index_select(0, expanded_return_idx)
        if attention_mask_col1 is not None:
            model_kwargs["attention_mask_col1"] = attention_mask_col1.index_select(0, expanded_return_idx)
        if attention_mask_col2 is not None:
            model_kwargs["attention_mask_col2"] = attention_mask_col2.index_select(0, expanded_return_idx)
        if attention_mask_col3 is not None:
            model_kwargs["attention_mask_col3"] = attention_mask_col3.index_select(0, expanded_return_idx)
        if attention_mask_col4 is not None:
            model_kwargs["attention_mask_col4"] = attention_mask_col4.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs_col0 is not None:
                encoder_outputs_col0["last_hidden_state"] = encoder_outputs_col0.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs_col0.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs_col0"] = encoder_outputs_col0

            if encoder_outputs_col1 is not None:
                encoder_outputs_col1["last_hidden_state"] = encoder_outputs_col1.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs_col1.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs_col1"] = encoder_outputs_col1


            if encoder_outputs_col2 is not None:
                encoder_outputs_col2["last_hidden_state"] = encoder_outputs_col2.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs_col2.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs_col2"] = encoder_outputs_col2


            if encoder_outputs_col3 is not None:
                encoder_outputs_col3["last_hidden_state"] = encoder_outputs_col3.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs_col3.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs_col3"] = encoder_outputs_col3


            if encoder_outputs_col4 is not None:
                encoder_outputs_col4["last_hidden_state"] = encoder_outputs_col4.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs_col4.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs_col4"] = encoder_outputs_col4

        return input_ids, model_kwargs


    def _prepare_attention_mask_for_generation(self, batch, model_kwargs):
        attention_mask_col0 = batch[1] if len(batch) >1 else None
        attention_mask_col1 = batch[3] if len(batch) >3 else None
        attention_mask_col2 = batch[5] if len(batch) >5 else None
        attention_mask_col3 = batch[7] if len(batch) >7 else None
        attention_mask_col4 = batch[9] if len(batch) >9 else None
        
        if not(attention_mask_col0 is None):
            model_kwargs["attention_mask_col0"] = attention_mask_col0
        if not(attention_mask_col1 is None):
            model_kwargs["attention_mask_col1"] = attention_mask_col1
        if not(attention_mask_col2 is None):
            model_kwargs["attention_mask_col2"] = attention_mask_col2
        if not(attention_mask_col3 is None):
            model_kwargs["attention_mask_col3"] = attention_mask_col3
        if not(attention_mask_col4 is None):
            model_kwargs["attention_mask_col4"] = attention_mask_col4

        return model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids_col0: torch.LongTensor,
        input_ids_col1: torch.LongTensor,
        input_ids_col2: torch.LongTensor,
        input_ids_col3: torch.LongTensor,
        input_ids_col4: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder_col0, encoder_col1, encoder_col2, encoder_col3, encoder_col4 = self.model.get_encoders()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            if not(input_ids_col0 is None):
                encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                attention_mask_col0 = encoder_kwargs.get("attention_mask_col0", None)
                encoder_outputs = encoder_kwargs.get('encoder_outputs_col0', None)
                if encoder_forward_stratergy == 'single':
                    model_kwargs["encoder_outputs_col0"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col0, encoder_outputs = encoder_outputs, input_ids = input_ids_col0, attention_mask = attention_mask_col0)
                else:
                    model_kwargs["encoder_outputs_col0"] , _ = self.model._loop_encoders(encoder_col0, encoder_outputs, input_ids_col0, attention_mask_col0, inc_count = 256)

            if encoder_forward_stratergy == 'single':
                 if not(input_ids_col1 is None):
                     encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                     attention_mask_col1 = encoder_kwargs.get("attention_mask_col1", None)
                     encoder_outputs = encoder_kwargs.get('encoder_outputs_col1', None)
                     model_kwargs["encoder_outputs_col1"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col1, encoder_outputs = encoder_outputs, input_ids = input_ids_col1, attention_mask = attention_mask_col1)

            if encoder_forward_stratergy == 'single':
                 if not(input_ids_col2 is None):
                     encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                     attention_mask_col2 = encoder_kwargs.get("attention_mask_col2", None)
                     encoder_outputs = encoder_kwargs.get('encoder_outputs_col2', None)
                     model_kwargs["encoder_outputs_col2"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col2, encoder_outputs = encoder_outputs, input_ids = input_ids_col2, attention_mask = attention_mask_col2)

            if encoder_forward_stratergy == 'single':
                 if not(input_ids_col3 is None):
                     encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                     attention_mask_col3 = encoder_kwargs.get("attention_mask_col0", None)
                     encoder_outputs = encoder_kwargs.get('encoder_outputs_col3', None)
                     model_kwargs["encoder_outputs_col3"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col3, encoder_outputs = encoder_outputs, input_ids = input_ids_col3, attention_mask = attention_mask_col3)

            if encoder_forward_stratergy == 'single':
                 if not(input_ids_col4 is None):
                     encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                     attention_mask_col0 = encoder_kwargs.get("attention_mask_col4", None)
                     encoder_outputs = encoder_kwargs.get('encoder_outputs_col4', None)
                     model_kwargs["encoder_outputs_col4"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col4, encoder_outputs = encoder_outputs, input_ids = input_ids_col4, attention_mask = attention_mask_col4)

        return model_kwargs
        
    def generate(self,
        batch,
        input_ids = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs, 
    ):
        #return_dict_in_generate = None
        #print("USE CACHE", use_cache)
        #use_cache = False
        use_cache = True
        input_ids_col0 = batch[0] if len(batch) >1 else None
        attention_mask_col0 = batch[1] if len(batch) >1 else None

        input_ids_col1 = batch[2] if len(batch) >3 else None
        attention_mask_col1 = batch[3] if len(batch) >3 else None

        input_ids_col2 = batch[4] if len(batch) >5 else None
        attention_mask_col2 = batch[5] if len(batch) >5 else None

        input_ids_col3 = batch[6] if len(batch) >7 else None
        attention_mask_col3 = batch[7] if len(batch) >7 else None

        input_ids_col4 = batch[8] if len(batch) >9 else None
        attention_mask_col4 = batch[9] if len(batch) >9 else None
    
        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.eos_token_id


        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs =  self._prepare_attention_mask_for_generation(
                batch, model_kwargs)

        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None
        input_list = [each for each in [input_ids_col0, input_ids_col1, input_ids_col2, input_ids_col3, input_ids_col4,] \
                            if not(each is None)]
        encoder_input_ids = torch.cat(input_list, 0)
        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids_col0, 
                                                                                input_ids_col1,
                                                                                input_ids_col2,
                                                                                input_ids_col3,
                                                                                input_ids_col4,
                                                                                 model_kwargs)

            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )



            
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        stopping_criteria = self._get_stopping_criteria(max_length=max_length, max_time=max_time)

        if is_greedy_gen_mode:
            #print("GREEDY SEARCHING")
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.model.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            #print("BEAM SEARCHING")
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            #print("BEAM SEARCH KWARGS", model_kwargs)
            return self.model.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


        elif is_group_beam_gen_mode:
            #print("GROUP BEAM SEARCHING")
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.model.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.model.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            return self.model.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


def sample_scorer(sample, model, tokenizer, nbeams, min_len, r_penalty, l_penalty, generator):
    references = []
    model_out = []
    rouge = Rouge()
    #generator = Data2TextGenerator(model, tokenizer)
    meteor_scores = []
    bleu_scores =[]
    for each in sample:
        outputs = generator.generate(each, num_beams = nbeams,  max_length = 400, min_length =min_len, repetition_penalty = r_penalty, length_penalty = l_penalty, encoder_forward_stratergy = 'loop', encoder_combination_type = 'addition')
        print("Outputs", outputs)
        model_output = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in outputs])
        target = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in each[-1]])
        if model_output.strip():
            print(model_output)
            print(target)
            print('=' * 13)
            model_out.append(model_output)
            references.append(target)
            #avg_len += first_batch[-1].shape[1]
            met_score = round(meteor_score.meteor_score([target], model_output), 4)
            meteor_scores.append(met_score)
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], model_output)
            bleu_scores.append(BLEUscore)
    print('='*13)
    print("Values: num_beam:%s || min_len:%s || r_penalty:%s || l_penalty:%s"%( nbeams, min_len, r_penalty, l_penalty))
    #print(avg_len/num_val)
    rougeScores = rouge.get_scores(model_out, references, avg=True)
    print("ROGUE", rougeScores)
    print("METEOR", sum(meteor_scores)/len(meteor_scores))
    print("BLEU", sum(bleu_scores)/len(bleu_scores))
    return references, model_out, rougeScores['rouge-1']['f'], rougeScores['rouge-l']['f']

def parameter_search(sample, model, tokenizer):

        num_beams_list = [3,5]
        min_lengths = [70,50,30,90]
        repetition_penalties = [1.0, 1.1, 1.2, 1.3]
        length_penalties = [1.0, 2.0, 3.0, 5.0]
        #rouge = Rouge()
        final_refs = []
        final_tgts = []
        max_rou1 = 0
        max_roul = 0
        final_num_beam = 0
        final_min_len = 0
        final_rpenalty = 0
        final_lpenalty =0 

        generator = Data2TextGenerator(model, tokenizer)
        for beam in num_beams_list:
                for min_len in min_lengths:
                    for r_penalty in repetition_penalties:
                        for l_penalty in length_penalties:
                            references, targets, rou1, roul = sample_scorer(sample, model, tokenizer, nbeams = beam, min_len = min_len, r_penalty = r_penalty, l_penalty = l_penalty, generator = generator)
                            if roul > max_roul:
                                final_num_beam = beam
                                final_min_len = min_len
                                final_rpenalty = r_penalty
                                final_lpenalty = l_penalty
                                max_roul = roul
        return final_num_beam, final_min_len, final_rpenalty, final_lpenalty

def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/sanjana', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    if data_type == 'robo':
        train_file = path + '/roboreviewer_summarization/data/%s'%(files[0])
        dev_file = path + '/roboreviewer_summarization/data/%s'%(files[1])
        test_file = path + '/roboreviewer_summarization/data/%s'%(files[2])

    elif data_type =='webnlg':
        train_file = path + '/roboreviewer_summarization/data/web_nlg_train.csv'
        dev_file = path + '/roboreviewer_summarization/data/web_nlg_dev.csv'
        test_file = path + '/roboreviewer_summarization/data/web_nlg_test.csv'

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1)
    summary_data.prepare_data()
    return summary_data


def run_sample_scorer(encoder_forward_stratergy = 'loop', encoder_combination_type = 'addition'):
    additional_special_tokens = ["<sep>", "<study>", "</study>",
            "<outcomes>", "</outcomes>",
            "<punchline_text>", "</punchline_text>",
            "<population>", "</population>",
            "<interventions>", "</interventions>",
            "<punchline_effect>", "</punchline_effect>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                    eos_token="</s>",
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens)
    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    hparams = argparse.Namespace()
    freeze_encoder = True
    freeze_embeds = True
    hparams.eval_beams = 4
    model_path = 'checkpoint_files/epoch=3-loss=0.39.ckpt'
    model = LitModel.load_from_checkpoint(checkpoint_path="/home/sanjana/roboreviewer_summarization/scripts/%s"%(model_path), encoder_forward_stratergy = encoder_forward_stratergy, encoder_combination_type = encoder_combination_type,)


    print("Loading data...")
    if encoder_forward_stratergy == 'loop':
        from Data2TextProcessor_loop import SummaryDataModule
        files = ['robo_train_linearized_per_study.csv', 
                            'robo_dev_linearized_per_study.csv', 'robo_test_linearized_per_study.csv']
        

    elif encoder_forward_stratergy == 'single':
        from Data2TextProcessor import SummaryDataModule
        files = ['robo_train_sep.csv', 
                            'robo_dev_sep.csv', 'robo_test_sep.csv']

    
    summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/sanjana', files = files)
    summary_data.setup("stage")
    val_data = summary_data.val_dataloader(data_type = 'robo')

    num_val = 50
    print("NUM EXAMPLES", num_val)
    it = iter(val_data)
    import random
    sample = random.sample(list(it), num_val)

    num_beams, min_len, repetition_penalty, length_penalty = parameter_search(sample, model, tokenizer)
    references, targets, _, _ = sample_score(list(it), model, tokenizer, num_beams, min_len, repetition_penalty, length_penalty)
    df_write = pd.DataFrame(list(zip(references, model_out)), columns=["Reference Summary", "Generated Summary"])
    file_name = '_'.join(model_path.split('/'))
    df_write.to_csv("%s.csv"%file_name)


        
if __name__ == '__main__':
    run_sample_scorer(encoder_forward_stratergy = 'loop', encoder_combination_type = 'addition')
