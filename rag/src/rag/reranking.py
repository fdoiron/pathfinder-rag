from typing import cast

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.config import Settings

DEFAULT_INSTRUCTION = (
    'Given a Pathfinder 1e rules query, identify the single canonical rules page that defines the specific '
    'spell, feat, condition, or creature type being asked about; not a class, domain, or category page that '
    'merely references it.'
)

# Qwen3-Reranker "Original Usage" scheme (requires transformers>=4.51.0): the checkpoint is a causal LM, not a
# SequenceClassification model so relevance is read off the next-token "yes"/"no" logits rather than a
# classification head. See https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#usage.
_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)
_PREFIX = f'<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n'
_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
_MAX_LENGTH = 1024  # ~450-token chunks + instruction/query/template overhead comfortably fit; throughput tradeoff


class RerankerUnavailableError(RuntimeError):
    """The reranking model could not be loaded."""


class LocalReranker:
    """Reranks via local model in-process, using Qwen3-Reranker's causal-LM yes/no next-token scoring."""

    def __init__(self, settings: Settings):
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device = torch.device('cuda')
        else:
            dtype = torch.float32
            device = torch.device('cpu')
        self._dtype = str(dtype)
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model, padding_side='left')
        model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(settings.reranker_model, dtype=dtype)
        self._model = model.eval().to(device)  # nn.Module.eval() is typed; PreTrainedModel's override isn't
        self._token_false_id = self._tokenizer.convert_tokens_to_ids('no')
        self._token_true_id = self._tokenizer.convert_tokens_to_ids('yes')
        self._prefix_tokens = self._tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)

    @property
    def torch_dtype(self) -> str:
        return self._dtype

    def rerank(self, query: str, texts: list[str], instruction: str | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0), dtype=np.float32)

        instruction = DEFAULT_INSTRUCTION if instruction is None else instruction
        pairs = [f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {text}' for text in texts]
        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)
        return np.array(scores, dtype=np.float32)

    def _process_inputs(self, pairs: list[str]) -> dict[str, torch.Tensor]:
        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=_MAX_LENGTH - len(self._prefix_tokens) - len(self._suffix_tokens),
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self._prefix_tokens + ele + self._suffix_tokens
        # max_length omitted here: individual sequences are already truncated above, and padding=True pads
        # dynamically to the batch's longest sequence. Passing max_length too just triggers a tokenizer warning.
        padded = self._tokenizer.pad(inputs, padding=True, return_tensors='pt')
        return {key: value.to(self._device) for key, value in padded.items()}

    @torch.no_grad()
    def _compute_logits(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        batch_scores = self._model(**inputs, logits_to_keep=1).logits[:, -1, :]
        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return cast('list[float]', log_probs[:, 1].exp().tolist())


def load_reranker(settings: Settings) -> LocalReranker:
    try:
        return LocalReranker(settings)
    except ValueError as e:  # unusable model. Malformed repo id (HFValidationError)
        raise RerankerUnavailableError(f'Cannot use reranking model {settings.reranker_model!r}: {e}') from e
    except OSError as e:  # hub and cache failures are OSErrors
        raise RerankerUnavailableError(
            f'Cannot load reranking model {settings.reranker_model!r}.Cause: {type(e).__name__}: {e}'
        ) from e
