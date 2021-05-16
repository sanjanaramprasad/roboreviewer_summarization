import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bart.configuration_bart import BartConfig
import torch
import torch.distributed as dist
from torch.nn import functional as Få
from BartForDataToTextGeneration import BartForDataToText
from transformers.generation_utils import GenerationMixin
from run_experiment import LitModel
from transformers import BartTokenizer
from Data2TextProcessor_1 import SummaryDataModule
import argparse

class Data2TextGenerator(GenerationMixin):

    def __init__(self, model, tokenizer):
        self.model = model.model 
        self.tokenizer = tokenizer 
        self.config = self.model.config
        self.device = self.model.device
        #print(self.config.max_length)

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
            print('ENC KWARGS', encoder_kwargs)
            if not(input_ids_col0 is None):
                model_kwargs["encoder_outputs_col0"]: ModelOutput = encoder_col0(input_ids_col0, return_dict=True, **encoder_kwargs)
            if not(input_ids_col1 is None):
                model_kwargs["encoder_outputs_col1"]: ModelOutput = encoder_col1(input_ids_col1, return_dict=True, **encoder_kwargs)
            if not(input_ids_col2 is None):
                model_kwargs["encoder_outputs_col2"]: ModelOutput = encoder_col2(input_ids_col2, return_dict=True, **encoder_kwargs)
            if not(input_ids_col3 is None):
                model_kwargs["encoder_outputs_col3"]: ModelOutput = encoder_col3(input_ids_col3, return_dict=True, **encoder_kwargs)
            if not(input_ids_col4 is None):
                model_kwargs["encoder_outputs_col4"]: ModelOutput = encoder_col4(input_ids_col4, return_dict=True, **encoder_kwargs)


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
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

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

            print("INPUT IDS", input_ids)
            print("MODEL KWARGS", model_kwargs)

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = LitModel.load_from_checkpoint(checkpoint_path="webnlg_sanity_model.ckpt")
    '''bart_model = BartForDataToText.from_pretrained('facebook/bart-base')    
    hparams = argparse.Namespace()

    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4

    model = LitModel(learning_rate = 1e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)'''
    summary_data = SummaryDataModule(tokenizer, data_files = ['/Users/sanjana/roboreviewer_summarization/data/web_nlg_train.csv', 
                                           '/Users/sanjana/roboreviewer_summarization/data/web_nlg_test.csv', 
                                           '/Users/sanjana/roboreviewer_summarization/data/web_nlg_dev.csv'], batch_size = 1)
    summary_data.prepare_data()
    summary_data.setup("stage")
    train_data = summary_data.train_dataloader()

    it = iter(train_data)
    first_batch = next(it)
    generator = Data2TextGenerator(model, tokenizer)

    generator.generate(first_batch)
