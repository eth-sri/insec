# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Perplexity Metric."""

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


_CITATION = """\

"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.

For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )

    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example 1:
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              add_start_token=False,
        ...                              predictions=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 0))
        647.0
        >>> print(round(results["perplexities"][0], 0))
        32.0

    Example 2:
        >>> from datasets import load_dataset
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:10] # doctest: +SKIP
        >>> input_texts = [s for s in input_texts if s!='']
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              predictions=input_texts)
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2)) # doctest: +SKIP
        576.76
        >>> print(round(results["perplexities"][0], 2)) # doctest: +SKIP
        889.28
"""


class Perplexity:

    def compute_all(self, encoded_predictions, model, tokenizer, batch_size: int = 1, add_start_token: bool = True, device=None, max_length=None):
        results = self.compute(encoded_predictions=encoded_predictions, model=model, tokenizer=tokenizer, device=model.device)["perplexities"]
        results_windowed = self.compute_windowed(encoded_predictions, model, tokenizer, batch_size=batch_size, add_start_token=add_start_token, device=device, max_length=max_length)
        return results, results_windowed

    def compute_windowed(self, full_prompt_tokens, model, tokenizer, batch_size: int = 1, add_start_token: bool = True, device=None, max_length=None):
        window_results = []

        full_prompt_tokens_windowed = full_prompt_tokens.unfold(1, 10, 10)
        for window in full_prompt_tokens_windowed[0]:
            window = window.unsqueeze(0)
            results = \
            self.compute(encoded_predictions=window, model=model, tokenizer=tokenizer, device=model.device)["perplexities"][0]
            window_results.append(results)
        return window_results

    def compute(
        self, encoded_predictions, model, tokenizer, batch_size: int = 1, add_start_token: bool = True, device=None, max_length=None
    ):

        encodings = encoded_predictions

        encoded_texts = encodings
        attn_masks = torch.ones_like(encoded_texts).to(device)

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
