import json
from collections import defaultdict
from typing import List, Callable
import logging
import os
import re
from subprocess import Popen, PIPE
from typing import List, Union


def get_vocab(train_data_path):
    vocab = set()
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            word = splits[1]
            vocab.add(word)
    return vocab


def get_word2id(train_data_path, do_lower_case=False, freq_threshold=2):
    word2count =defaultdict(int)
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            word = splits[1]
            if do_lower_case:
                word = word.lower()
            word2count[word] += 1

    word2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2}
    index = 3
    for word, count in word2count.items():
        if count >= freq_threshold:
            word2id[word] = index
            index += 1

    return word2id


def get_label_list(train_data_path, labels_to_add: List[str] = ['[CLS]', '[SEP]']):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            dep = splits[7]
            if dep not in label_list:
                label_list.append(dep)

    label_list.extend(labels_to_add)
    return label_list


def get_pos2id(train_data_path):
    pos2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2}
    index = 3
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            pos = splits[3]
            if pos not in pos2id:
                pos2id[pos] = index
                index += 1
    return pos2id


def get_wordpair_list(data_path):
    wordpair2id = {}
    i = 1
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            splits = line.split()
            wordpair2id[(splits[0], splits[1])] = i
            i += 1
    return wordpair2id


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)

def check_model_type_in(model_name: str, candidates: Union[str, List[Union[str, re.Pattern]]]):
    if isinstance(candidates, str):
        candidates = [candidates]
    
    for each in candidates:
        if isinstance(each, str) and each in model_name:
            return True
        if isinstance(each, re.Pattern) and re.search(each, model_name):
            return True
    
    return False

def is_local_main_process():
    return int(os.environ.get('LOCAL_RANK', -1)) == 0

def is_main_process():
    return int(os.environ.get('RANK', -1)) == 0

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', -1))

def get_rank():
    return int(os.environ.get('RANK', -1))

def get_gpu_usage(message: str = None):
    p = Popen(['nvidia-smi'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    res = []
    if p.returncode != 0:
        return None
    for line in stdout.decode().split('\n'):
        if 'MiB' in line and '%' in line:
            res.append(line)
    
    if message is not None:
        res = message.split('\n') + res

    return '\n'.join(res)

def move_to_device(*inputs, device: str = "cpu"):
    res = []
    for each in inputs:
        if hasattr(each, 'to'):
            res.append(each.to(device))
        else:
            res.append(each)
    
    return res

class DummyLogger:
    def __init__(self) -> None:
        pass

for name in [each for each in dir(logging.Logger) if each[:2] != '__']:
    if isinstance(getattr(logging.Logger, name), Callable):
        setattr(DummyLogger, name, lambda self, *args, **kwargs: None)