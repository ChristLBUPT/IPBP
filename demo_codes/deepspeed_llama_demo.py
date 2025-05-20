import deepspeed
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
from transformers import LlamaTokenizer, LlamaForCausalLM
import datasets

def load_model_tokenizer(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def load_data():

def main():
    model, tokenizer = load_model_tokenizer('../../pretrained-models/open_llama_3b_v2')
