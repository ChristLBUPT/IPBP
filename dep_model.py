from __future__ import absolute_import, division, print_function

import copy
import math
import random as rd

import numpy as np
import torch
from torch import nn, tensor, Tensor
from torch.utils.data import Dataset
from modules import ZenModel, Biaffine, MLP, LayerNormalization, PositionalEncoding, LlamaTokenizer, OpenELMModel, LlamaModel, BertModel
from modules.graph_utils.message_passing_demo import ModifiedGINEConv
from transformers import BertTokenizer, GemmaTokenizer, GemmaModel
from peft import LoraConfig, LoraModel
from transformers_xlnet import XLNetModel, XLNetTokenizer
from util import eisner, ZenNgramDict, ispunct
from dep_helper import save_json, load_json, check_model_type_in
import json
import os
import re
import subprocess
from dataclasses import dataclass, Field, fields
import logging
from typing import List, Dict, Tuple, Union
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data, Batch


IMPOSSIBLE=-1145141919810
def hparam_field(*, default=IMPOSSIBLE, default_factory=IMPOSSIBLE, init=True, repr=True,
          hash=None, compare=True, **metadata):
    if default != IMPOSSIBLE and default_factory != IMPOSSIBLE:
        raise ValueError('cannot specify both default and default_factory')

    return Field(default, default_factory, init, repr, hash, compare,
                 metadata)
    

@dataclass
class Hparam:
    max_seq_length: int = hparam_field(default=128, help= 
            "The maximum total input sequence length after WordPiece tokenization. "
            "Sequences longer than this will be truncated, and sequences shorter "
            "than this will be padded."
        )
    use_bert: bool = hparam_field(default=False, help="Whether to use BERT.")
    use_llm: bool = hparam_field(default=False, help="Whether to use LLMs (Gemma, open-llama, OpenELM, OPT, etc).") 
    lora_llm: bool = hparam_field(default=False, help="Whether or not enable lora on llm")
    lora_config: str = hparam_field(default=None, help="Path to a config json or a config json of huggingface PEFT `LoraConfig`")
    use_xlnet: bool = hparam_field(default=False, help="Whether to use XLNet.")
    use_zen: bool = hparam_field(default=False, help="Whether to use ZEN.")
    do_lower_case: bool = hparam_field(default=False, help="Set this flag if you are using an uncased model.")
    mlp_dropout: float = hparam_field(default=0.33, help="Dropout of pre-bi-affine attention linear layer")
    n_mlp_arc: int = hparam_field(default=500, help="Output dimension of the pre-bi-affine linear layer of arcs")
    n_mlp_rel: int = hparam_field(default=100, help="Output dimension of the pre-bi-affine linear layer of relations")
    use_biaffine: bool = hparam_field(default=False, 
        help="Whether or not use bi-affine attention module to calculate edge existance and edge relation scores")
    use_gcn: bool = hparam_field(default=False, help="Whether of not use attention GCNs")
    gcn_attn_only: bool = hparam_field(default=False, help="Whether or not only pass concatenated attention scores to GCN, "
        "otherwise will pass all-0s as node embeddings")
    gcn_involve_node_emb: bool = hparam_field(default=False, help="Whether or not fuse edge embeddings in the final layer output of GCN")
    gcn_transpose_edge_scores: bool = hparam_field(default=False, 
        help="Whether or not use the attention dot-product score of i as q and j as k to be the score of i being the head (or being which kind of head) of j")
    use_edge_gcn: bool = hparam_field(default=False, help="Whether or not use gcn with a topological structure organized by edge connections")
    use_aoa: bool = hparam_field(default=False, help="whether or not use attention network to process attention of each layer")
    use_pos: bool = hparam_field(default=False, help="Whether or not to have POS(Part-of-Speech) emebddings added to token embeddings")
    use_encoder: bool = hparam_field(default=False, help="Whether or not to have extra transformer encoding layers on top of base model")
    num_layers: int = hparam_field(default=3, help="Number of encoder layers")

    def __getitem__(self, key):
        if key in vars(self):
            return vars(self)[key]
    
    # def save_json(self, path: str) -> str:
            
# DEFAULT_HPARA = {
#     'max_seq_length': 128,
#     'use_bert': False,
#     'use_llm': False,
#     'lora_llm': False,
#     'lora_config': None,
#     'use_xlnet': False,
#     'use_zen': False,
#     'do_lower_case': False,
#     'use_pos': False,
#     'mlp_dropout': 0.33,
#     'n_mlp_arc': 500,
#     'n_mlp_rel': 100,
#     'use_biaffine': True,
#     #
#     'use_encoder': False,
#     'num_layers': 3,
#     'd_model': 1024,
#     'num_heads': 8,
#     'd_ff': 2048,
# }


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, head=None, label=None, pos=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.head = head
        self.label = label
        self.pos = pos
    
    def __repr__(self) -> str:
        return \
"""
class InputExample {{
    guid = {guid};
    text_a = {text_a};
    text_b = {text_b};
    head = {head};
    label = {label};
    pos = {pos};
}}""".format(**vars(self))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, head_idx, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 word_ids=None, pos_ids=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.head_idx = head_idx
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.word_ids = word_ids
        self.pos_ids = pos_ids

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

# class ExampleDataset(Dataset):
#     def __init__(self, training_examples: List[InputExample]):
#         self.training_examples = training_examples
    
#     def __getitem__(self, index):
#         return self.training_examples[index]
    
#     def __len__(self):
#         return len(self.training_examples)
     

class DependencyParser(nn.Module):
    def __init__(
        self, 
        labelmap: Dict[str, int],
        hpara: Hparam, 
        model_path: str, 
        word2id: Dict[str, int] = None, 
        pos2id: Dict[str, int] = None, 
        from_pretrained: bool = True, 
        model_dtype: Union[torch.dtype, str] = torch.bfloat16,
        device_to_place: Union[str, int] = None,
        logger: logging.Logger = None
    ):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1 # 
        self.arc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.rel_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.logger = logger

        if hpara.use_zen:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.llm = None
        self.xlnet = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara.do_lower_case)
            if from_pretrained:
                self.log(f'initializing pre-trained BERT from `{model_path}`')
                self.bert = BertModel.from_pretrained(model_path, cache_dir='', device_map='auto' if device_to_place is None else device_to_place, torch_dtype=model_dtype)
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            self.hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara.use_llm:
            if check_model_type_in(model_path, ['ELM', re.compile(r'[Ll]lama')]):
                tokenizer_cls = LlamaTokenizer
                if 'ELM' in model_path:
                    model_cls = OpenELMModel
                elif re.search(re.compile(r'[Ll]lama'), model_path):
                    model_cls = LlamaModel
            else:
                raise NotImplementedError(f'error, cannot inference model type from `{model_path}`')
            self.tokenizer = tokenizer_cls.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.log(f'setting `pad_token` to `eos_token([{self.tokenizer.eos_token_id}]: {self.tokenizer.eos_token})` because it is null')
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.log(f'initializing pre-trained LLM from `{model_path}`')
            self.llm = model_cls.from_pretrained(model_path, device_map='auto' if device_to_place is None else device_to_place, torch_dtype=model_dtype)
            if hasattr(self.llm, 'base_model'): # use base model
                self.llm = self.llm.base_model
            self.llm_type = self.llm.__class__.__name__
            self.hidden_size = self.llm.config.hidden_size if hasattr(self.llm.config, 'hidden_size') else self.llm.config.model_dim
            # init lora model
            if self.hpara.lora_llm:
                if self.hpara.lora_config is not None:
                    if re.search("{.*}", self.hpara.lora_config):
                        try:
                            lora_config = json.loads(self.hpara.lora_config)
                        except json.JSONDecodeError as e:
                            print(f"error, cannot decode '{self.hpara.lora_config}'")
                            raise e
                    else:
                        lora_config = json.load(open(self.hpara.lora_config, 'r'))

                else:
                    lora_config = {}
                
                lora_config = LoraConfig(**lora_config)
                self.llm = LoraModel(self.llm, lora_config, 'default_lora')
            self.dropout = nn.Dropout(0) # we use a dropout probability of 0 since gemma use dropout of probability 0 by default 

        elif self.hpara.use_xlnet:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara.do_lower_case)
            if from_pretrained:
                self.xlnet = XLNetModel.from_pretrained(model_path)
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                key_list = list(state_dict.keys())
                reload = False
                for key in key_list:
                    if key.find('xlnet.') > -1:
                        reload = True
                        state_dict[key[key.find('xlnet.') + len('xlnet.'):]] = state_dict[key]
                    state_dict.pop(key)
                if reload:
                    self.xlnet.load_state_dict(state_dict)
            else:
                config, model_kwargs = XLNetModel.config_class.from_pretrained(model_path, return_unused_kwargs=True)
                self.xlnet = XLNetModel(config)

            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
        elif self.hpara.use_zen:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara.do_lower_case)
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()
        
        hidden_size = self.hidden_size

        if self.hpara.use_pos:
            assert pos2id is not None, f"Error, `use_pos` requires `pos2id`"
            self.pos2id = pos2id
            if self.hpara.use_encoder:
                self.pos_embedding = nn.Embedding(len(self.pos2id), int(self.hpara.d_model))
            else:
                self.pos_embedding = nn.Embedding(len(self.pos2id), hidden_size)
                self.layer_norm_encoder = LayerNormalization(hidden_size)
                self.pos_norm = LayerNormalization(hidden_size)
        else:
            self.pos2id = None
            self.pos_embedding = None

        if self.hpara.use_encoder:
            self.linear = MLP(n_in=hidden_size, n_hidden=(self.hpara.d_model), dropout=0.2)
            self.linear_dep = MLP(n_in=hidden_size, n_hidden=(self.hpara.d_model), dropout=0.2)
            self.layer_norm_encoder = LayerNormalization(self.hpara.d_model)
            self.layer_norm_dep = LayerNormalization(self.hpara.d_model)
            self.positional_embedding = PositionalEncoding(self.hpara.d_model, dropout=0.2)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hpara.d_model, nhead=self.hpara.num_heads,
                                                       dim_feedforward=self.hpara.d_ff)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.hpara.num_layers)

            hidden_size = self.hpara.d_model
        else:
            self.linear_dep = None
            self.layer_norm_dep = None
            self.positional_embedding = None
            self.encoder = None

        self.word2id = None

        if self.hpara.use_gcn:
            if self.hpara.use_llm:
                if 'OpenELM' in self.llm_type:
                    self.gcn_edge_dim = sum(self.llm.config.num_query_heads)
                elif 'Llama' in self.llm_type:
                    self.gcn_edge_dim = self.llm.config.num_attention_heads * self.llm.config.num_hidden_layers
                else:
                    raise ValueError(f"Unsupported LLM type {self.llm_type}")
            elif self.hpara.use_bert:
                self.gcn_edge_dim = self.bert.config.num_attention_heads * self.bert.config.num_hidden_layers
            else:
                raise ValueError(f"gcn is only applicable for BERT and LLM")
            self.gcn = ModifiedGINEConv(self.hidden_size, self.gcn_edge_dim)
            self.edge_classifier = nn.Linear(2 * self.hpara.n_mlp_arc + self.gcn_edge_dim if self.hpara.gcn_involve_node_emb else self.gcn_edge_dim, 1)
            self.relation_classifier = nn.Linear(2 * self.hpara.n_mlp_rel + self.gcn_edge_dim if self.hpara.gcn_involve_node_emb else self.gcn_edge_dim, self.num_labels)
            if self.hpara.gcn_involve_node_emb:
                self.gcn_mlp_arc_h = MLP(n_in=hidden_size,
                                    n_hidden=self.hpara.n_mlp_arc,
                                    dropout=self.hpara.mlp_dropout)
                self.gcn_mlp_arc_d = MLP(n_in=hidden_size,
                                    n_hidden=self.hpara.n_mlp_arc,
                                    dropout=self.hpara.mlp_dropout)
                self.gcn_mlp_rel_h = MLP(n_in=hidden_size,
                                    n_hidden=self.hpara.n_mlp_rel,
                                    dropout=self.hpara.mlp_dropout)
                self.gcn_mlp_rel_d = MLP(n_in=hidden_size,
                                    n_hidden=self.hpara.n_mlp_rel,
                                    dropout=self.hpara.mlp_dropout)

        elif self.hpara.use_edge_gcn:
            raise NotImplementedError("edge_gcn is not implementd yet")

        elif self.hpara.use_aoa:
            raise NotImplementedError("aoa is not implementd yet")

        elif self.hpara.use_biaffine:
            self.mlp_arc_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara.n_mlp_arc,
                                 dropout=self.hpara.mlp_dropout)
            self.mlp_arc_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara.n_mlp_arc,
                                 dropout=self.hpara.mlp_dropout)
            self.mlp_rel_h = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara.n_mlp_rel,
                                 dropout=self.hpara.mlp_dropout)
            self.mlp_rel_d = MLP(n_in=hidden_size,
                                 n_hidden=self.hpara.n_mlp_rel,
                                 dropout=self.hpara.mlp_dropout)

            self.arc_attn = Biaffine(n_in=self.hpara.n_mlp_arc,
                                     bias_x=True,
                                     bias_y=False)
            self.rel_attn = Biaffine(n_in=self.hpara.n_mlp_rel,
                                     n_out=self.num_labels,
                                     bias_x=True,
                                     bias_y=True)
        else:
            self.linear_arc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.rel_classifier_1 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.rel_classifier_2 = nn.Linear(hidden_size, self.num_labels, bias=False)
            self.bias = nn.Parameter(torch.tensor(self.num_labels, dtype=torch.float), requires_grad=True)
            nn.init.zeros_(self.bias)
    
    def log(self, message, level: int = logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                word_ids=None, pos_ids=None,
                input_ngram_ids=None, ngram_position_matrix=None, return_hidden_states_and_attentions=False, extract_hidden_states_and_attentions=False,
                ):

        if self.bert is not None:
            # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            if return_hidden_states_and_attentions or extract_hidden_states_and_attentions:
                output = self.bert.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)
                sequence_output = output.last_hidden_state
            else:
                sequence_output= self.bert.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        
        elif self.llm is not None:
            if self.hpara.use_aoa or self.hpara.use_gcn or self.hpara.use_edge_gcn:
                output = self.llm.forward(input_ids=input_ids, attention_mask=attention_mask, output_attention_queries=True)
                sequence_output = output.last_hidden_state

        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()
        
        # for idx, each in enumerate(input_ids):
        #     if any([each.find('##') >=0 for each in self.tokenizer.convert_ids_to_tokens(each)]):
        #         breakpoint()
        #         break

        # `valid` indicates which tokens are the first subtoken of a word
        # `attention_mask_label` is the `attention_mask` of whole-words
        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        sent_lens = []
        for i in range(batch_size):
            # get the hidden states of the first subtoken of every token
            # TODO: ??? replace loop assignment to `torch.where(valid_ids==1, sequence_output, valid_output)`
            try:
                temp = sequence_output[i][valid_ids[i] == 1]
            except Exception:
                breakpoint()
            sent_len = attention_mask_label[i].sum()
            sent_lens.append(sent_len.item())
            # valid_output[i][:temp.size(0)] = temp
            # valid pads `1`s for 
            valid_output[i][:sent_len] = temp[:sent_len]
        
        if extract_hidden_states_and_attentions:
            return valid_output.detach().cpu(), [each.detach().cpu() for each in output.hidden_states], [each.detach().cpu() for each in output.attentions]

        if self.encoder is not None:
            encoder_input = self.linear(valid_output)
            if self.pos_embedding is not None:
                pos_embedding = self.pos_embedding(pos_ids)
                encoder_input = encoder_input + pos_embedding
            encoder_input = self.layer_norm_encoder(encoder_input)
            encoder_input = self.positional_embedding(encoder_input)
            encoder_feature = self.encoder(encoder_input)
            valid_output = self.layer_norm_dep(encoder_feature)

        elif self.pos_embedding is not None:
            pos_embedding = self.pos_embedding(pos_ids)
            valid_output = self.layer_norm_encoder(valid_output)
            pos_embedding = self.pos_norm(pos_embedding)
            valid_output = pos_embedding + valid_output
        
        if self.hpara.use_gcn or self.hpara.use_edge_gcn or self.hpara.use_aoa:
            # recalculate (from queries and keys) or direct extract attentions from outputs
            if self.hpara.use_llm:
                attn_scores = ()
                for layer_idx in range(len(output.queries)):
                    k = output.past_key_values[layer_idx][0]
                    q = output.queries[layer_idx]
                    if q.shape[-3] != k.shape[-3]:
                        n_groups = q.shape[-3] / k.shape[-3]
                        assert n_groups == int(n_groups)
                        n_groups = int(n_groups)
                    else:
                        n_groups = 1
                    
                    k = torch.repeat_interleave(k, n_groups, -3)
                    this_attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(sequence_output.shape[-1])
                    attn_scores += (this_attn_score,)
                
                attentions = torch.cat(attn_scores, 1)
            else:
                attentions = torch.cat(output.attentions, 1) 

        if self.hpara.use_gcn:
            # size: [batch_size, num_heads * num_layers, seq_length, seq_length]
            data_list = []
            
            for i in range(batch_size):
                valid_mask = valid_ids[i] == 1
                sent_len = sent_lens[i]
                x = valid_output[i][:sent_len]
                valid_attention = attentions[i][:, valid_mask][:, :, valid_mask][:, :sent_len, :sent_len].contiguous()
                # size [num_heads * num_layers, seq_length, seq_length]
                edge_attr = valid_attention.view(self.gcn_edge_dim, sent_len * sent_len).permute(1, 0)
                edge_index = torch.stack([torch.tensor([[i] * sent_len for i in range(sent_len)]).view(-1), torch.arange(sent_len).repeat(sent_len)]).to(edge_attr.device)
                data_list.append(Data(
                    x=torch.zeros_like(x) if self.hpara.gcn_attn_only else x, 
                    edge_index=edge_index, edge_attr=edge_attr
                ))
            
            batch = Batch.from_data_list(data_list)
            
            node_features, edge_features = self.gcn(batch.x, batch.edge_index, batch.edge_attr, )
            s_arc = torch.ones(batch_size, max_len, max_len, device=edge_features.device) * -114514 
            s_rel = torch.ones(batch_size, max_len, max_len, self.num_labels, device=edge_features.device) * -114514 
            if self.hpara.gcn_involve_node_emb:
                x_j = node_features[batch.edge_index[0]]
                x_i = node_features[batch.edge_index[1]]
                s_arc_flatten = self.edge_classifier(torch.cat([self.gcn_mlp_arc_d(x_i), edge_features, self.gcn_mlp_arc_h(x_j)], dim=-1))
                s_rel_flatten = self.relation_classifier(torch.cat([self.gcn_mlp_rel_d(x_i), edge_features, self.gcn_mlp_rel_h(x_j)], dim=-1))
            
            else:
                s_arc_flatten = self.edge_classifier(edge_features)
                s_rel_flatten = self.relation_classifier(edge_features)
            lb = 0
            for sample_idx in range(batch_size):
                sent_len = sent_lens[sample_idx]
                s_arc[sample_idx][:sent_len, :sent_len] = s_arc_flatten[lb: lb + sent_len * sent_len].view(sent_len, sent_len)
                s_rel[sample_idx][:sent_len, :sent_len] = s_rel_flatten[lb: lb + sent_len * sent_len].view(sent_len, sent_len, -1)
                if self.hpara.gcn_transpose_edge_scores:
                    s_arc[sample_idx][:sent_len, :sent_len].transpose_(-2, -1)
                    s_rel[sample_idx][:sent_len, :sent_len].transpose_(-2, -1)
                # for i in range(sent_len):
                #     for j in range(sent_len):
                #         s_arc[sample_idx][i][j] = s_arc_flatten[lb + i * sent_len + j].item()
                #         s_rel[sample_idx][i][j] = s_rel_flatten[lb + i * sent_len + j]
                
                lb += sent_len * sent_len
            
            s_arc = s_arc.contiguous()
            s_rel = s_rel.contiguous()


        elif self.hpara.use_biaffine:
            valid_output = self.dropout(valid_output)

            arc_h = self.mlp_arc_h(valid_output)
            arc_d = self.mlp_arc_d(valid_output)
            rel_h = self.mlp_rel_h(valid_output)
            rel_d = self.mlp_rel_d(valid_output)

            # get arc and rel scores from the bilinear attention
            # [batch_size, seq_len, seq_len]
            s_arc = self.arc_attn(arc_d, arc_h)
            # [batch_size, seq_len, seq_len, n_rels]
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))
        else:
            tmp_arc = self.linear_arc(valid_output).permute(0, 2, 1)
            s_arc = torch.bmm(valid_output, tmp_arc)

            # [batch_size, seq_len, seq_len, n_rels]
            rel_1 = self.rel_classifier_1(valid_output)
            rel_2 = self.rel_classifier_2(valid_output)
            rel_1 = torch.stack([rel_1] * max_len, dim=1)
            rel_2 = torch.stack([rel_2] * max_len, dim=2)
            s_rel = rel_1 + rel_2 + self.bias
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~attention_mask_label.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @staticmethod
    def init_hyper_parameters(args):
        hpara_field_names = [each.name for each in fields(Hparam)]
        init_dict = {}
        for attr_name in dir(args):
            if attr_name in hpara_field_names:
                init_dict[attr_name] = getattr(args, attr_name)

        hyper_parameters = Hparam(**init_dict)
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()
    
    def save_labelmap(self, output_dir: str):
        save_json(output_dir, self.labelmap)
    
    def save_hpara(self, output_dir: str):
        json_obj = {}
        for each in fields(Hparam):
            json_obj[each.name] = getattr(self.hpara, each.name)
        
        json.dump(json_obj, open(output_dir, 'w'))

    def save_model(self, output_dir, vocab_dir):

        if self.bert or self.llm:
            self.tokenizer.save_pretrained(output_dir)
            model = self.bert or self.llm
            model.save_pretrained(output_dir)

        # output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        # torch.save(self.state_dict(), output_model_path)

        self.save_labelmap(os.path.join(output_dir, 'labelset.json'))

        # output_hpara_file = os.path.join(output_dir, 'hpara.json')
        # save_json(output_hpara_file, self.hpara)
        self.save_hpara(os.path.join(output_dir, 'hpara.json'))


        if self.pos2id is not None:
            output_hpara_file = os.path.join(output_dir, 'pos2id.json')
            save_json(output_hpara_file, self.pos2id)

        if self.word2id is not None:
            output_hpara_file = os.path.join(output_dir, 'word2id.json')
            save_json(output_hpara_file, self.word2id)

        # output_config_file = os.path.join(output_dir, 'config.json')
        # with open(output_config_file, "w", encoding='utf-8') as writer:
        #     if self.bert:
        #         writer.write(self.bert.config.to_json_string())
        #     elif self.xlnet:
        #         writer.write(self.xlnet.config.to_json_string())
        #     elif self.zen:
        #         writer.write(self.zen.config.to_json_string())
        # output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
        # command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        # subprocess.run(command, shell=True)

        # if self.bert:
        #     vocab_name = 'vocab.txt'
        # elif self.xlnet:
        #     vocab_name = 'spiece.model'
        # elif self.zen:
        #     vocab_name = 'vocab.txt'
        # else:
        #     raise ValueError()
        # vocab_path = os.path.join(vocab_dir, vocab_name)
        # command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
        # subprocess.run(command, shell=True)

        # if self.zen:
        #     ngram_name = 'ngram.txt'
        #     ngram_path = os.path.join(vocab_dir, ngram_name)
        #     command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
        #     subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        pos_file = os.path.join(model_path, 'pos2id.json')
        if os.path.exists(pos_file):
            pos2id = load_json(pos_file)
        else:
            pos2id = None

        word_file = os.path.join(model_path, 'word2id.json')
        if os.path.exists(word_file):
            word2id = load_json(word_file)
        else:
            word2id = None

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, word2id=word2id, pos2id=pos2id)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res

    @staticmethod
    def set_not_grad(module):
        for para in module.parameters():
            para.requires_grad = False

    def load_data(self, data_path, do_predict=False, longest_samples_first=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        lines = readfile(data_path, flag)

        examples = self.process_data(lines, flag)

        if longest_samples_first:
            examples.sort(key=lambda example: len(example.text_a.split()), reverse=True)
            examples_longest = examples[:10]
            examples_rest = examples[10:]
            rd.shuffle(examples_rest)
            examples = examples_longest + examples_rest

        return examples

    @staticmethod
    def process_data(lines, flag):
        data = []
        for sentence, head, label, pos in lines:
            data.append((sentence, head, label, pos))
        examples = []
        for i, (sentence, head, label, pos) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, head=head,
                                         label=label, pos=pos))
        return examples

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = self.arc_criterion(arc_scores, arcs)
        rel_loss = self.rel_criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    @staticmethod
    def decode(arc_scores, rel_scores, mask):
        arc_preds = eisner(arc_scores, mask)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def convert_examples_to_features(self, examples: List[InputExample]):
        """convert a batch of InputExamples to a batch of InputFeatures"""

        features = []

        length_list = []  # length of tokenized sentences
        tokens_list = []  # tokenized sentences
        head_idx_list = []# head of each word
        labels_list = []
        valid_list = []
        label_mask_list = []
        punctuation_idx_list = []

        pos_list = []

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            head_list = example.head
            tokens = []
            head_idx = []
            labels = []
            valid = []
            label_mask = []

            punctuation_idx = []

            poslist = example.pos

            if len(textlist) > self.hpara.max_seq_length - 2: # truncation by #words (not #subword-tokens)
                textlist = textlist[:self.hpara.max_seq_length - 2]
                labellist = labellist[:self.hpara.max_seq_length - 2]
                head_list = head_list[:self.hpara.max_seq_length - 2]
                poslist = poslist[:self.hpara.max_seq_length - 2]

            if self.hpara.use_llm:
                if re.search(r'Gemma|ELM|Llama', self.llm_type):
                    blank = chr(0x2581)
                else:
                    blank = 'Ġ'
                sentence = ''
                for i, word in enumerate(textlist):
                    if ispunct(word):
                        if i >= 1 and ispunct(textlist[i - 1][-1]): 
                            sentence += f' {word}'# prevent two consequent puncts be tokenized into one subword, e.g. the u.s., -> ['_the', '_u', '.', 's', '.,']
                        else:
                            sentence += word
                        punctuation_idx.append(i + 1) # i + 1 because `<bos>` will be added before tokens
                    else:
                        sentence += f' {word}'

                tokens = self.tokenizer.tokenize(sentence)
                word_ptr = 0
                char_ptr = 0
                for token in tokens:
                    token_wo_blank = token.replace(blank, '')
                    if textlist[word_ptr][char_ptr: char_ptr + len(token_wo_blank)] == token_wo_blank:
                        if char_ptr + len(token_wo_blank) == len(textlist[word_ptr]):
                            valid.append(1)
                            head_idx.append(head_list[word_ptr])
                            labels.append(labellist[word_ptr])
                            label_mask.append(1)
                            word_ptr += 1
                            char_ptr = 0
                        else:
                            valid.append(0)
                            char_ptr += len(token_wo_blank)
                    
                    else:
                        # breakpoint()
                        raise ValueError(f'{token =}, {tokens =}, {textlist =}')
            # end `if self.hpara.use_llm`
            else:
                for i, word in enumerate(textlist):
                    if ispunct(word):
                        punctuation_idx.append(i+1)
                    # if self.hparam['use_llm']:
                    #     subtokens = self.tokenizer.tokenize(f'<bos> {word}')
                    #     if subtokens[0] == '<bos>' and subtokens[1].replace(chr(0x2581, '')) in word, f'word: {word}'
                    #         raise UserWarning(f'error while word {word}, subtokens: {subtokens}')
                    
                    # else:
                    subtokens = self.tokenizer.tokenize(word)
                    tokens.extend(subtokens)
                    label_1 = labellist[i]
                    for m in range(len(subtokens)):
                        # if (self.hpara.use_bert and m == 0) or (self.hpara.use_llm and m == len(subtokens) - 1):
                        if m == 0:
                            valid.append(1)
                            head_idx.append(head_list[i])
                            labels.append(label_1)
                            label_mask.append(1)
                        else:
                            valid.append(0)
            # end if model is BERT

            length_list.append(len(tokens))
            tokens_list.append(tokens)
            assert head_idx == head_list, f'{head_idx = } != {head_list = }'
            head_idx_list.append(head_idx)
            assert labels == labellist, f'{labels = } != {labellist = }'
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            punctuation_idx_list.append(punctuation_idx)

            pos_list.append(poslist)
        # end for example in examples

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + (1 if self.hpara.use_llm else 2) # BERT has [CLS] and [SEP] while llm don't add eos tokens by default
        label_pad_length = max(label_len_list) + 1 # only start tokens correspond to a label

        if self.llm is not None:
            if 'OPT' in self.llm_type:
                start_token = "</s>"
            elif 'Gemma' in self.llm_type:
                start_token = "<bos>"
            elif re.search(r'Llama|ELM', self.llm_type):
                start_token = "<s>"
            else:
                raise ValueError(f'LLM type `{self.llm_type}` not supported')

        else:
            start_token = "[CLS]"

        for indx, (example, tokens, head_idxs, labels, valid, label_mask, punctuation_idx, pos) in \
                enumerate(zip(examples, tokens_list, head_idx_list,
                              labels_list, valid_list, label_mask_list, punctuation_idx_list, pos_list)):

            ntokens = []
            segment_ids = []
            label_ids = []
            head_idx = []

            ntokens.append(start_token)
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            head_idx.append(-1)
            label_ids.append(self.labelmap[start_token])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
                head_idx.append(head_idxs[i])

            if self.bert is not None:
                ntokens.append("[SEP]")
                segment_ids.append(0)
                valid.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                # pad `input_ids`, `input_mask`, and `segment_ids` with 0, pad `valid` with 1
                input_ids.append(self.tokenizer.pad_token_id)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                head_idx.append(-1)
                label_ids.append(0)
                label_mask.append(0)

            eval_mask = copy.deepcopy(label_mask)
            eval_mask[0] = 0
            # ignore all punctuation if not specified
            for idx in punctuation_idx:
                if idx < label_pad_length:
                    eval_mask[idx] = 0

            if self.pos_embedding is not None:
                pos_ids = [self.pos2id['[CLS]']]
                for i in range(len(pos)):
                    if pos[i] in self.pos2id:
                        pos_ids.append(self.pos2id[pos[i]])
                    else:
                        pos_ids.append(self.pos2id['<UNK>'])
                while len(pos_ids) < label_pad_length:
                    pos_ids.append(0)
                assert len(pos_ids) == label_pad_length
            else:
                pos_ids = None

            word_ids = None

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(head_idx) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.hpara.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            # if sum(label_mask) != sum(valid):
            #     breakpoint()
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              head_idx=head_idx,
                              label_id=label_ids,
                              valid_ids=valid, 
                              # valid_ids means which tokens are start/end of subtokens #
                              # (in order to make subword-tokenized tokens(input_ids) match label_ids)
                              label_mask=label_mask,
                              # label_mask means after padding, which `label_id`s are real `label_id`s (not pad ids)
                              # and it will be used to calculate loss
                              eval_mask=eval_mask,
                              # eval_mask are similar to label_mask except:
                              # 1. the first token ([CLS] for BERT or <bos> for Gemma) is set to 0
                              # 2. all punctuaion token are set to 0
                              word_ids=word_ids,
                              pos_ids=pos_ids,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              ))
        return features

    def feature2input(self, device, feature):
        # try:
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        # except TypeError as e:
        #     print(feature[0].input_ids)
        #     raise e

        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_head_idx = torch.tensor([f.head_idx for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        head_idx = all_head_idx.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None

        if self.pos_embedding is not None:
            all_pos_ids = torch.tensor([f.pos_ids for f in feature], dtype=torch.long)
            pos_ids = all_pos_ids.to(device)
        else:
            pos_ids = None

        word_ids = None

        return input_ids, input_mask, l_mask, eval_mask, head_idx, label_ids, \
               word_ids, pos_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids




def readfile(filename, flag) -> List[Tuple[List[str], List[int], List[str], List[str]]]:
    """
    read dataset file and return tuple to form instances
    Returns:
        Tuples of [sentence, head_indices, labels, POS]
    """
    data = []
    sentence = []
    head = []
    label = []
    pos = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '':
                    if len(sentence) > 0:
                        data.append((sentence, head, label, pos))
                        sentence = []
                        head = []
                        label = []
                        pos = []
                    continue
                splits = line.split('\t')
                sentence.append(splits[1])
                pos.append(splits[3])
                head.append(int(splits[6]))
                label.append(splits[7])
            if len(sentence) > 0:
                data.append((sentence, head, label, pos))
        else:
            raise ValueError()
            # for line in lines:
            #     line = line.strip()
            #     if line == '':
            #         continue
            #     label_list = ['NN' for _ in range(len(line))]
            #     data.append((line, label_list))
    return data
