from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple
import torch
from dataclasses import dataclass

@dataclass
class BaseModelOutputWithPastAndQueries(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding) and queries 
    to restore attention scores after forward pass, since LLMs often use attention implementations that dont't return attention scores, 
    such as pytorch2.0's `scaled_dot_product_attention`.

    queries (`tuple(torch.FloatTensor)`, *optional*, returned only when `output_attention_queries=True` is passed:
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
    """
    queries: Optional[Tuple[torch.FloatTensor]] = None