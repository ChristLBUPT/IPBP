from transformers import LlamaTokenizer, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList
from dep_model import DependencyParser, Hparam
import json
import argparse
import fire
from typing import Dict, Tuple
import torch
import numpy as np
import accelerate
import os

class StopOnTripleGrave(StoppingCriteria):
    def __init__(self, tokenizer: LlamaTokenizer):
        super().__init__()
        self._stop = False
        self.tokenizer = tokenizer
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0][-1])
        if "```" in decoded_text:
            return True
        else:
            return False

# accelerate.utils.set_seed()

def load_pretrained_model_and_tokenizer(model_path: str) -> Tuple[LlamaForCausalLM, DependencyParser]:
    dep_parser_llama = DependencyParser(
        json.load(open('./configs/openllama/sample_labelmap.json')), 
        Hparam(**json.load(open('./configs/openllama/sample_hpara.json'))), 
        model_path, device_to_place='cuda:0'
    ) 
    del dep_parser_llama.llm
    model = LlamaForCausalLM.from_pretrained(model_path)
    return model, dep_parser_llama

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    model_path: str = '../pretrained-models/open_llama_3b_v2/', 
    prompt_name: str = 'with_examples',
    max_length: int = 4096, length_penalty: int = 1.0, num_beams: int = 1,
    seed: int = 114514, device: str = 'cuda:0', n_sumples: int = 10,
):
    seed_everything(seed)
    model, dep_parser = load_pretrained_model_and_tokenizer(model_path)
    model = model.to(device)
    tokenizer: LlamaTokenizer = dep_parser.tokenizer
    data = dep_parser.load_data('./data/dev.conllu')
    prompts: Dict[str, str] = {} 
    #json.load(open('./configs/prompts/prompts.json'))
    for prompt_name in os.listdir('./configs/prompts/'):
        if prompt_name.endswith('.prompt'):
            prompts.update({prompt_name: open(f'./configs/prompts/{prompt_name}').read()})
    prompt_format = prompts[prompt_name]
    print(f'prompt length : {len(tokenizer.tokenize(prompt_format))}')
    for sample_idx, sample in enumerate(data):
        if sample_idx == n_sumples: 
            break
        print(f"sample: {sample_idx:03d}".center(80, '='))
        sentence = prompt_format.format(**{"sent": sample.text_a})
        output = model.generate(**tokenizer(
            sentence, return_tensors='pt').to(device), max_length=max_length, num_beams=num_beams, length_penalty=length_penalty, #stopping_criteria=StoppingCriteriaList([StopOnTripleGrave(tokenizer)])
        )
        # print('input'.center(60, '='))
        # print(sentence)
        print('output'.center(60, '='))
        print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])

if __name__ == '__main__':
    fire.Fire(main)
