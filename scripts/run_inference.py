import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn import functional as Få

from transformers.generation_utils import GenerationMixin
from run_experiment import LitModel
from transformers import BartTokenizer
class Data2TextGenerator(GenerationMixin):

    def __init__(self, model, tokenizer):
        self.model = model 
        self.tokenizer = tokenizer 


    def generate(self,
        input_ids: Optional[torch.LongTensor] = None,
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

        max_length = max_length if max_length is not None else self.model.model.config.max_length
        num_beams = num_beams if num_beams is not None else self.model.model.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.model.model.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.model.model.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.model.model.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.model.model.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.model.model.config.eos_token_id


        output_scores = output_scores if output_scores is not None else self.model.model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.model.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.model.model.config.return_dict_in_generate
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

        encoder_input_ids = input_ids if self.model.model.config.is_encoder_decoder else None


if __name__ == '__main__':
    model = LitModel.load_from_checkpoint(checkpoint_path="webnlg_sanity_model.ckpt")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    generator = Data2TextGenerator(model, tokenizer)
    generator.generate()