import torch
from torch import nn, tensor, Tensor
from typing import List, Iterable, Tuple, Literal
import os
import os.path as osp
import re
import click
from jsonargparse import ArgumentParser, CLI
import torch.utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from line_profiler import profile
import json
import math


def formalize_pickle_file_name(original_save_name: str, data_proportion: str = '1', transpose: bool = False, save_idx: int = None, num_saves: int = None):
    file_name, file_ext = original_save_name.rsplit('.', maxsplit=1)
    file_name = f"{file_name}_{str(data_proportion).replace('.', '_')}{'_transpose' if transpose else ''}"
    if save_idx is not None and num_saves is not None:
        # pad save_idx with 0, making it have the save length as num_saves
        file_name = f"{file_name}_{str(save_idx).zfill(len(str(num_saves)))}of{num_saves}"
    return f'{file_name}.{file_ext}'

def get_possible_num_chunks(kde_save_pth: str, data_proportion: str = '1', transpose: bool = False):
    pickle_file_names = [
        *filter(lambda x: re.match(f".*_attn_samples_{data_proportion}{'_transpose' if transpose else ''}.*of.*.pkl", x), os.listdir(kde_save_pth))
    ]
    possible_total_num_chunks = set([re.search(r'.*_(\d+)of(\d+)', each).group(2) for each in pickle_file_names])
    if len(possible_total_num_chunks) != 1:
        raise RuntimeError(f"found multiple possible total number of attn-sample save chunks: {', '.join(possible_total_num_chunks)}")
    total_num_chunks = int(list(possible_total_num_chunks)[0])
    return total_num_chunks


def truncate_neg_samples(input_dir: str, output_dir: str, data_proportion: str = '1', transpose: bool = False):
    num_chunks = get_possible_num_chunks(input_dir, data_proportion, transpose)
    os.makedirs(output_dir, exist_ok=True)
    for current_num_save in tqdm(range(num_chunks), desc='truncating neg samples...'):
        this_neg_features: Tensor = torch.load(osp.join(input_dir, formalize_pickle_file_name('neg_attn_samples.pkl', data_proportion, transpose, current_num_save, num_chunks)), map_location='cpu')
        this_label_features: Tuple[Tensor] = torch.load(osp.join(input_dir, formalize_pickle_file_name('label_attn_samples.pkl', data_proportion, transpose, current_num_save, num_chunks)), map_location='cpu')
        total_pos_sample_cnt = sum([each.shape[0] for each in this_label_features])
        selected_neg_sample_indices = torch.Tensor(range(0, this_neg_features.shape[0], int(this_neg_features.shape[0] / total_pos_sample_cnt))).long()
        torch.save(this_neg_features[selected_neg_sample_indices], 
            osp.join(output_dir, formalize_pickle_file_name('neg_attn_samples.pkl', data_proportion, transpose, current_num_save, num_chunks)))
        torch.save(this_label_features, osp.join(output_dir, formalize_pickle_file_name('label_attn_samples.pkl', data_proportion, transpose, current_num_save, num_chunks)))

class SimpleMLP(nn.Module):
    def __init__(self, n_input: int, hidden_sizes: List[int], n_labels: int):
        super().__init__()
        if len(hidden_sizes) >= 1:
            self.mlp = nn.Sequential(
                nn.Linear(n_input, hidden_sizes[0]),
                *[nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)],
                nn.Linear(hidden_sizes[-1], n_labels)
            )
        else:
            self.mlp = nn.Linear(n_input, n_labels)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class IndependentMLP(nn.Module):
    def __init__(self, n_input: int, hidden_sizes: List[int], n_labels: int, hidden_activation: Literal['tanh', 'relu', 'sigmoid', 'leaky_relu'] = 'tanh'):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.middle_weights = nn.ParameterList()
        hidden_sizes = [1] + hidden_sizes
        if len(hidden_sizes) >= 1:
            for hidden_layer_idx, hidden_size in enumerate(hidden_sizes[1:], start=1):
                self.middle_weights.append(nn.Parameter(torch.randn(n_input, hidden_sizes[hidden_layer_idx - 1], hidden_size)))
            self.final_layer = nn.Parameter(torch.randn(n_input, hidden_sizes[-1], n_labels))
    
        else:
            self.final_layer = nn.Parameter(torch.randn(n_input, 1, n_labels))
        self.hidden_activation = hidden_activation
        self.hidden_function_map = {"tanh": torch.tanh, 'relu': torch.relu, 'sigmoid': torch.sigmoid, 'leaky_relu': torch.nn.functional.leaky_relu}
    
    def forward(self, x: Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1) # [B, n_features, 1, 1]
        for hidden_layer_idx, hidden_layer in enumerate(self.middle_weights):
            x = torch.matmul(x, hidden_layer)
            x = self.hidden_function_map[self.hidden_activation](x)
        return torch.matmul(x, self.final_layer)

    
class ChunkCache:
    def __init__(self, train_data_dir: str, data_proportion: str = '1', transpose: bool = False, max_chunks_in_memory: int = 3):
        self.train_data_dir = train_data_dir
        self.data_proportion = data_proportion
        self.transpose = transpose
        self.max_chunks_in_memory = max_chunks_in_memory
        self.total_num_chunks = get_possible_num_chunks(train_data_dir, data_proportion, transpose)
        self.current_chunk_indices = []
        self.current_features = []
        self.current_labels = []
    
    def move_to_front(self, index):
        self.current_chunk_indices.insert(0, self.current_chunk_indices.pop(index))
        self.current_features.insert(0, self.current_features.pop(index))
        self.current_labels.insert(0, self.current_labels.pop(index))
    
    def load_features(self, chunk_idx):
        if not isinstance(chunk_idx, int):
            chunk_idx = chunk_idx.item()
        neg_features: Tensor = torch.load(osp.join(self.train_data_dir, formalize_pickle_file_name('neg_attn_samples.pkl', self.data_proportion, self.transpose, chunk_idx, self.total_num_chunks)), map_location='cpu')
        label_features: Tuple[Tensor] = torch.load(osp.join(self.train_data_dir, formalize_pickle_file_name('label_attn_samples.pkl', self.data_proportion, self.transpose, chunk_idx, self.total_num_chunks)), map_location='cpu')
        return torch.cat([torch.cat(label_features), neg_features], dim=0), \
            torch.cat([
                torch.cat([
                    torch.ones(label_features[label_idx].shape[0]) * (label_idx + 1) \
                        for label_idx in range(len(label_features))
                    ]), 
                torch.zeros(neg_features.shape[0]).long()
            ])
        
    def __getitem__(self, index):
        if index not in self.current_chunk_indices:
            if len(self.current_chunk_indices) >= self.max_chunks_in_memory:
                self.current_chunk_indices.pop()
                self.current_features.pop()
                self.current_labels.pop()
            self.current_chunk_indices.insert(0, index)
            features, labels = self.load_features(index)
            self.current_features.insert(0, features)
            self.current_labels.insert(0, labels)
        else:
            self.move_to_front(self.current_chunk_indices.index(index))
        return self.current_features[0], self.current_labels[0]


class FeatureDataset:
    def __init__(self, train_data_dir: str, data_proportion: str = '1', transpose: bool = False, batch_size: int = 1024, max_chunks_in_memory: int = 3):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.num_chunks = get_possible_num_chunks(train_data_dir, data_proportion, transpose)
        self.metadata_filename = osp.join(train_data_dir, formalize_pickle_file_name('attn_samples_metadata.json', data_proportion, transpose))
        if not osp.exists(self.metadata_filename):
            print(f'generating metadata file {self.metadata_filename}...')
            self.metadata = form_chunked_dataset_metadata(train_data_dir, data_proportion, transpose)
            with open(self.metadata_filename, 'w') as f:
                json.dump(self.metadata, f)
        else:
            with open(self.metadata_filename, 'r') as f:
                self.metadata = json.load(f)
        self.chunk_starts = torch.LongTensor([0] + self.metadata['num_samples']).cumsum(0)
        self.dimension: int = self.metadata['feature_dimension'] 
        self.num_labels: int = self.metadata['num_labels']
        self.num_samples: int = self.chunk_starts[-1].item()
        self.batch_size = batch_size
        self.max_chunks_in_memory = max_chunks_in_memory
        self.num_batches = math.ceil(self.num_samples / batch_size)
        self.chunk_cache = ChunkCache(train_data_dir, data_proportion, transpose, self.max_chunks_in_memory)
    
    def __getitem__(self, index):
        if index >= self.num_batches:
            raise StopIteration
        batch_start = self.batch_size * index
        batch_end = min(self.batch_size * (index + 1) - 1, self.num_samples - 1)
        start_chunk_idx = torch.searchsorted(self.chunk_starts, batch_start, right=True) - 1
        start_inner_idx = batch_start - self.chunk_starts[start_chunk_idx]
        end_chunk_idx = torch.searchsorted(self.chunk_starts, batch_end, right=True) - 1
        end_inner_idx = batch_end - self.chunk_starts[end_chunk_idx]

        if start_chunk_idx == end_chunk_idx:
            this_samples, this_labels = self.chunk_cache[start_chunk_idx]
            return this_samples[start_inner_idx: end_inner_idx + 1], this_labels[start_inner_idx: end_inner_idx + 1]
        else:
            samples, labels = [], []
            for chunk_idx in range(start_chunk_idx, end_chunk_idx + 1):
                this_samples, this_labels = self.chunk_cache[chunk_idx]
                if chunk_idx == start_chunk_idx:
                    samples.append(this_samples[start_inner_idx:])
                    labels.append(this_labels[start_inner_idx:])
                elif chunk_idx == end_chunk_idx:
                    samples.append(this_samples[:end_inner_idx + 1])
                    labels.append(this_labels[:end_inner_idx + 1])
                else:
                    samples.append(this_samples)
                    labels.append(this_labels)
            
            return torch.cat(samples, dim=0), torch.cat(labels, dim=0)
    
    def __len__(self):
        return self.num_batches

    
def load_data(kde_save_path: str, data_proportion: float = 0.2, transpose: bool = False):
    pos_features: Tensor  = torch.load(os.path.join(kde_save_path, formalize_pickle_file_name('pos_attn_samples.pkl', data_proportion, transpose)))
    neg_features: Tensor = torch.load(os.path.join(kde_save_path, formalize_pickle_file_name('neg_attn_samples.pkl', data_proportion, transpose)))
    label_features: Iterable[Tensor] = torch.load(os.path.join(kde_save_path, formalize_pickle_file_name('label_attn_samples.pkl', data_proportion, transpose)))
    label_features_notnull = []
    unmasked_indices = []
    for idx, feature in enumerate(label_features):
        if feature.shape[0] != 0:
            label_features_notnull.append(feature)
            unmasked_indices.append(idx)
    return pos_features, neg_features, label_features_notnull, unmasked_indices

# def concatenate_data(pos_features: Tensor, neg_features: Tensor, label_features: Iterable[Tensor], with_labels: bool = False):
#     # torch.random.manual_seed()
#     if not with_labels:
#         mixed_training_features = torch.cat([pos_features, neg_features], dim=0)
#         labels = torch.cat([torch.ones(pos_features.shape[0]).long(), torch.zeros(neg_features.shape[0]).long()])
#         indices = torch.randperm(mixed_training_features.shape[0])
#         mixed_training_features, labels  = mixed_training_features[indices], labels[indices]
#         return mixed_training_features, labels
#     else:
#         mixed_training_features = torch.cat([torch.cat(label_features), neg_features], dim=0)
#         labels = torch.cat([torch.cat([torch.ones(label_features[label_idx].shape[0]) * (label_idx + 1) for label_idx in range(len(label_features))]), torch.zeros(neg_features.shape[0]).long()])
#         indices = torch.randperm(mixed_training_features.shape[0])
#         mixed_training_features, labels  = mixed_training_features[indices], labels[indices]
#         return mixed_training_features, labels

def form_chunked_dataset_metadata(kde_save_pth: str, data_proportion: str = '1', transpose: bool = False):
    total_num_chunks = get_possible_num_chunks(kde_save_pth, data_proportion, transpose)
    num_samples = []
    num_labels = -1
    feature_dimension = None
    for current_num_save in (pbar := trange(total_num_chunks, desc='forming metadata...')):
        this_neg_features: Tensor = torch.load(osp.join(kde_save_pth, formalize_pickle_file_name('neg_attn_samples.pkl', data_proportion, transpose, current_num_save, total_num_chunks)), map_location='cpu')
        this_label_features: Tuple[Tensor] = torch.load(osp.join(kde_save_pth, formalize_pickle_file_name('label_attn_samples.pkl', data_proportion, transpose, current_num_save, total_num_chunks)), map_location='cpu')
        if set([each.shape[1] for each in this_label_features]) != {this_neg_features.shape[1]}:
            raise RuntimeError(f"error, found different feature dimensions: (label) {set([each.shape[1] for each in this_label_features])} vs. (neg) {this_neg_features.shape[1]}")
        if feature_dimension is None:
            feature_dimension = this_neg_features.shape[1]
        elif feature_dimension != this_neg_features.shape[1]:
            raise RuntimeError(f"error, found different feature dimensions from different chunks: {feature_dimension} vs. {this_neg_features.shape[1]}")
        num_samples.append(this_neg_features.shape[0] + sum([each.shape[0] for each in this_label_features]))
        for idx, each in enumerate(this_label_features):
            if each.shape[0] != 0:
                num_labels = max(num_labels, idx + 2) # idx + 2 includes the `0`(neg) label
                pbar.set_postfix({'num_samples': f'[..., {sum(num_samples)}]', 'num_labels': num_labels})
    
    return {"num_samples": num_samples, "num_labels": num_labels, "feature_dimension": feature_dimension}
            



def load_chunked_data_and_shuffle(kde_save_pth: str, num_chunks: int):
    total_num_chunks = get_possible_num_chunks(kde_save_pth)
    mixed_training_features, labels = [], []
    chunk_indices = torch.randperm(total_num_chunks).tolist() # randomly select chunks
    for current_num_save in tqdm(chunk_indices[:num_chunks], desc='loading attention samples and shuffling...'):
        this_neg_features: Tensor = torch.load(osp.join(kde_save_pth, formalize_pickle_file_name('neg_attn_samples.pkl', 1, False, current_num_save, total_num_chunks)), map_location='cpu')
        this_label_features: Tuple[Tensor] = torch.load(osp.join(kde_save_pth, formalize_pickle_file_name('label_attn_samples.pkl', 1, False, current_num_save, total_num_chunks)), map_location='cpu')
        this_mixed_training_features = torch.cat([torch.cat(this_label_features), this_neg_features], dim=0) # concatenate every label and negative features
        this_labels = torch.cat([torch.cat([torch.ones(this_label_features[label_idx].shape[0]) * (label_idx + 1) for label_idx in range(len(this_label_features))]), torch.zeros(this_neg_features.shape[0]).long()])
        # [1, 1, 1, ..., 2, 2, 2, ... n_labels, n_labels, ..., 0, 0, 0, ...(many)]
        indices = torch.randperm(this_mixed_training_features.shape[0])
        this_mixed_training_features, this_labels = this_mixed_training_features[indices], this_labels[indices]
        mixed_training_features.append(this_mixed_training_features)
        labels.append(this_labels)
    
    return mixed_training_features, labels
    # return FeatureDataset(mixed_training_features, labels)

# @click.command()
# @click.option('--train_data_dir', type=str, default='../pretrained-models/kde/open_llama_3b_v2/')
# @click.option('--val_data_dir', type=str, default='../pretrained-models/kde/open_llama_3b_v2/val')
# @click.option('--force_recache', is_flag=True)
# @click.option('--num_epochs', type=int, default=128)
# @click.option('--batch_size', type=int, default=16384)
# @click.option('--data_proportion', type=float, default=0.2)
# @click.option('--transpose', type=bool, default=False)
# @click.option('--device', type=str, default='cuda:1')
# @click.option('--with_labels', is_flag=True)
@profile
def mlp(
    # data args
    train_data_dir: str = '/home/user/pretrained-models/kde/open_llama_7b/',
    val_data_dir: str = '/home/user/pretrained-models/kde/open_llama_7b_debug',
    # force_recache: bool = False,
    # data_proportion: float = 0.2,
    # model args
    with_labels: bool = True,
    hidden_sizes: List[int] = [], # `-1` means no hidden layer (linear layer)
    # training args
    num_epochs: int = 128,
    batch_size: int = 65536,
    device: str = 'cpu',
    seed: int = 114514,
    l1_lambda: float = 1e-5,
    l2_lambda: float = 1e-5,
    # save_directory = '/home/user/pretrained-models/explaination_models/elasticnet/',
    # utilities args
    num_chunks: int = 1919810,
    verbose: bool = False
):
    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # get num_threads
    print('torch num_threads:', torch.get_num_threads(), 'torch num_interop_threads:', torch.get_num_interop_threads())
    # pos_features, neg_features, label_features, unmasked_indices = load_data(train_data_dir, data_proportion, transpose)
    # pos_features_val, neg_features_val, label_features_val, unmasked_indices_val = load_data(val_data_dir, 1, transpose)
    # cache_pth = os.path.expanduser('~/.dataset_cache')
    # if os.path.exists(cache_pth) and not force_recache:
    #     print(f'loading cached data from `{cache_pth}`..')
    #     loaded_cache = torch.load(cache_pth)
    #     mixed_training_features, training_labels, mixed_val_features, val_labels = loaded_cache
    # else:
    def formalize_hidden_sizes():
        return 'h_' + ('_'.join(map(str, hidden_sizes)) if len(hidden_sizes) != 0 else 'null')
    print(f'loading and reshuffling training and validation data...')
    # mixed_training_features, training_labels = load_chunked_data_and_shuffle(train_data_dir, num_chunks)
    training_dataset = FeatureDataset(train_data_dir, '1', False, batch_size)
    # mixed_val_features, val_labels = load_chunked_data_and_shuffle(val_data_dir, num_chunks)
    val_dataset = FeatureDataset(val_data_dir, '1', False, batch_size)
    save_directory = osp.join(train_data_dir, 'baselines', f'mlp_{formalize_hidden_sizes()}_{l1_lambda}_{l2_lambda}')
    os.makedirs(save_directory, exist_ok=True)
    print(f'checkpoints and logs will be saved to {save_directory}')
        # print(f'caching data to `{cache_pth}`...')
        # torch.save([mixed_training_features, training_labels, mixed_val_features, val_labels], cache_pth)
    # dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=4)
    # dl_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=4)

    print(f'initializing model and optimizer...')
    # set all seeds
    assert with_labels, f'only `with_labels=True` is supported now'

    module = SimpleMLP(training_dataset.dimension, hidden_sizes, training_dataset.num_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=0)
    # mix training data and shuffle
    # concatenated_data_cache_dir = '/tmp/explaination_baselines'
    # os.makedirs(concatenated_data_cache_dir, exist_ok=True)
    # concatenated_data_cache_path = os.path.join(concatenated_data_cache_dir, 'mixed_training_features.pkl')
    # if not force_recache and os.path.exists(concatenated_data_cache_path):
    #     print(f'loading cached training data from {concatenated_data_cache_path}...')
    #     mixed_training_features, training_labels = torch.load(concatenated_data_cache_path)
    # else:
    #     print(f'concatenating training data...')
    #     mixed_training_features, training_labels = concatenate_data(pos_features, neg_features, label_features, with_labels)
    #     torch.save([mixed_training_features, training_labels], concatenated_data_cache_path)
    # mixed_val_features, val_labels = concatenate_data(pos_features_val, neg_features_val, label_features_val, with_labels)

    module = module.to(device).to(torch.bfloat16)

    best_arc_acc = 0
    # flush and create training log
    open(osp.join(save_directory, 'log.jsonl'), 'w').close()

    for epoch in range(num_epochs):
        print('=' * 80)
        print(f'Epoch {epoch:3d} {formalize_hidden_sizes()} λ_1 {l1_lambda} λ_2 {l2_lambda}'.center(80))
        print('=' * 80)
        for x, y in (pbar := tqdm(training_dataset)):
            print_if_verbose(f'fetched data, moving to device')
            x, y = x.to(device), y.to(device).long()
            print_if_verbose(f'zero grad')
            optimizer.zero_grad()
            print_if_verbose(f'forward')
            y_pred = module(x)
            print_if_verbose(f'calculating loss')
            loss = criterion(y_pred, y) + l1_lambda * sum([param.abs().sum() for param in module.parameters()]) + l2_lambda * sum([(param ** 2).sum() for param in module.parameters()])
            print_if_verbose(f'backwarding')
            loss.backward()
            print_if_verbose(f'optimizing parameters')
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        # print(f'Epoch {epoch} loss: {loss.item()}')
        # evaluate and calculate acc
        n_corrects = 0
        arc_corrects = 0
        num_arcs = 0
        print('validating...')
        for x, y in val_dataset:
            x, y = x.to(device), y.to(device)
            num_arcs += (y != 0).long().sum().item()
            y_pred = module(x)
            correct_mask = (y_pred.argmax(dim=1) == y)
            n_corrects += (correct_mask).sum().item()
            arc_corrects += (correct_mask)[y != 0].sum().item()
        
        this_acc, this_arc_acc = n_corrects / val_dataset.num_samples, arc_corrects / num_arcs
        epoch_metrics_json = {'acc': this_acc, 'arc_acc': this_arc_acc, 'epoch': epoch}
        with open(osp.join(save_directory, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(epoch_metrics_json) + '\n')

        print(f'Epoch {epoch} acc: {this_acc} arc acc: {this_arc_acc}')
        if this_arc_acc > best_arc_acc:
            best_arc_acc = this_arc_acc
            print(f'new best arc acc: {best_arc_acc}, saving model...')
            torch.save(module.state_dict(), osp.join(save_directory, 'best_model.pt'))
            json.dump(epoch_metrics_json, open(osp.join(save_directory, 'metrics.json'), 'w'))


def independent_mlp(
    # data args
    train_data_dir: str = '/home/user/pretrained-models/kde/open_llama_7b/',
    val_data_dir: str = '/home/user/pretrained-models/kde/open_llama_7b_dev',
    # model args
    hidden_sizes: List[int] = [2], # `-1` means no hidden layer (linear layer)
    hidden_activation: Literal['tanh', 'relu', 'sigmoid', 'leaky_relu'] = 'tanh',
    # training args
    num_epochs: int = 12,
    batch_size: int = 2048,
    device: str = 'cuda:0',
    seed: int = 114514,
    lr: float = 0.01,
    l1_lambda: float = 1e-5,
    l2_lambda: float = 1e-5,
    # save_directory = '/home/user/pretrained-models/explaination_models/elasticnet/',
    # utilities args
    # num_chunks: int = 1919810,
    verbose: bool = False
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # get num_threads
    print('torch num_threads:', torch.get_num_threads(), 'torch num_interop_threads:', torch.get_num_interop_threads())
    print(f'loading and reshuffling training and validation data...')
    # mixed_training_features, training_labels = load_chunked_data_and_shuffle(train_data_dir, num_chunks)
    training_dataset = FeatureDataset(train_data_dir, '1', False, batch_size)
    # mixed_val_features, val_labels = load_chunked_data_and_shuffle(val_data_dir, num_chunks)
    val_dataset = FeatureDataset(val_data_dir, '1', False, batch_size)
    def formalize_hidden_sizes_and_activation():
        return 'h_' + ('_'.join(map(str, hidden_sizes)) if len(hidden_sizes) != 0 else 'null') + f'_{hidden_activation}'
    save_directory = osp.join(train_data_dir, 'baselines', f'independent_mlp_{formalize_hidden_sizes_and_activation()}_{l1_lambda}_{l2_lambda}')
    if 'balance' in train_data_dir:
        save_directory += '_balanced'
    os.makedirs(save_directory, exist_ok=True)
    print(f'checkpoints and logs will be saved to {save_directory}')
    # flush and create training log
    open(osp.join(save_directory, 'log.jsonl'), 'w').close()

    print(f"initializing model (hidden_size: {', '.join(map(str, hidden_sizes))}; activation: {hidden_activation}) and optimizer...")
    # set all seeds

    module = IndependentMLP(training_dataset.dimension, hidden_sizes, training_dataset.num_labels, hidden_activation,)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=0)
    lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, len(training_dataset)) # first epoch warmup
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    module = module.to(device)#.to(torch.bfloat16)

    best_arc_acc = 0

    for epoch in range(num_epochs):
        print('=' * 80)
        print(f'Epoch {epoch:3d} ({hidden_sizes} {hidden_activation} {l1_lambda} {l2_lambda})'.center(80))
        print('=' * 80)
        for x, y in (pbar := tqdm(training_dataset)):
            x, y = x.to(device).float(), y.to(device).long()
            # breakpoint()
            if epoch == 0:
                lr_scheduler_warmup.step()
            optimizer.zero_grad()
            y_pred = module.forward(x) # [B, n_features, 1, n_labels]
            y_pred.squeeze_(-2)
            B, n_features, n_labels = y_pred.shape
            y_pred = y_pred.view(B * n_features, n_labels)
            y = y.unsqueeze(-1).expand(B, n_features).reshape(-1)
            result_loss = criterion(y_pred, y)
            l1_reg = l1_lambda * sum([param.abs().sum() for param in module.parameters()]) if l1_lambda != 0 else Tensor([0]).to(y_pred.device)
            l2_reg = l2_lambda * sum([(param ** 2).sum() for param in module.parameters()]) if l2_lambda != 0 else Tensor([0]).to(y_pred.device)
            loss = result_loss + l1_reg + l2_reg
            loss.backward()
            with torch.no_grad():
                grad_sum = 0
                num_params = 0
                for each in module.parameters():
                    grad_sum += each.grad.abs().sum()
                    num_params += each.numel()

            optimizer.step()
            pbar.set_postfix({'loss': loss.item(), 'l_res': result_loss.item(), 'l1': l1_reg.item(), 'l2': l2_reg.item(), 'lr': optimizer.param_groups[0]['lr']})
                             #'mean_grad': (grad_sum / num_params).item()})
        if epoch != 0:
            lr_scheduler.step()
        # print(f'Epoch {epoch} loss: {loss.item()}')
        # evaluate and calculate acc
        n_corrects = 0
        arc_corrects = 0
        num_arcs = 0
        total_num_arcs = 0
        print('validating...')
        for x, y in val_dataset:
            x, y = x.to(device).float(), y.to(device)
            y = y.unsqueeze(-1).expand(-1, n_features) # [B, n_features]
            num_arcs += (y != 0).long().sum().item()
            total_num_arcs += y.numel()
            y_pred = module(x).squeeze(-2) # [B, n_features, n_bales]
            correct_mask = (y_pred.argmax(dim=-1) == y)
            n_corrects += (correct_mask).sum().item()
            arc_corrects += (correct_mask)[y != 0].sum().item()
        
        this_acc, this_arc_acc = n_corrects / total_num_arcs, arc_corrects / num_arcs
        epoch_metrics_json = {'acc': this_acc, 'arc_acc': this_arc_acc, 'epoch': epoch}
        with open(osp.join(save_directory, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(epoch_metrics_json) + '\n')
        print(f'Epoch {epoch} acc: {this_acc} arc acc: {this_arc_acc}')
        if this_arc_acc > best_arc_acc:
            best_arc_acc = this_arc_acc
            print(f'new best arc acc: {best_arc_acc}, saving model...')
            torch.save(module.state_dict(), osp.join(save_directory, 'best_model.pt'))
            json.dump(epoch_metrics_json, open(osp.join(save_directory, 'metrics.json'), 'w'))

@torch.no_grad()
def v_information(
    train_data_dir: str = "../pretrained-models/kde/open_llama_7b", 
    model_pth: str = "../pretrained-models/kde/open_llama_7b_balanced/baselines/independent_mlp_h_2_4_leaky_relu_0.0_0.0_balanced/best_model.pt", 
    hidden_sizes: List[int] = [2, 4], 
    hidden_activation: Literal['relu', 'tanh', 'leaky_relu', 'sigmoid'] = 'leaky_relu', batch_size: int = 32, device: str = 'cuda:0', save_every: int = 128):
    training_dataset = FeatureDataset(train_data_dir, '1', False, batch_size=batch_size)

    v_information = torch.zeros(training_dataset.dimension, training_dataset.num_labels).float().to(device)
    module = IndependentMLP(training_dataset.dimension, hidden_sizes, training_dataset.num_labels, hidden_activation,)
    module.load_state_dict(torch.load(model_pth))
    module.eval()
    v_information_save_dir = osp.join(osp.split(model_pth)[0], 'v_information.pt')
    print(f'v_information will be saved to {v_information_save_dir}')
    module = module.to(device)
    B, NF, L = batch_size, training_dataset.dimension, training_dataset.num_labels
    for data_idx, (x, y) in enumerate(pbar := tqdm(training_dataset)):
        x, y = x.to(device).float(), y.to(device).long()
        y_pred = module.forward(x).squeeze(-2).sigmoid() # [batch_size, n_features, n_labels]
        label_predictions = y_pred.gather(-1, y.unsqueeze(-1).expand(B, NF).unsqueeze(-1))
        mask_index = torch.zeros_like(y_pred)
        for idx, this_y in enumerate(y):
            mask_index[idx, :, this_y] = 1
        label_neg_predictions = (1 - y_pred)
        label_neg_predictions[mask_index.bool()] = 1
        for prediction, neg_prediction, this_y in zip(label_predictions, label_neg_predictions, y): #prediction: [NF, 1]
            # if (torch.log2(prediction) <= -1e12).any():
            #     breakpoint()

            v_information[:, this_y] -= torch.log2(prediction).squeeze(-1)
            # neg_prediction = torch.cat([neg_prediction[:, :this_y], torch.ones(NF, 1).to(neg_prediction.device), neg_prediction[:, this_y + 1:]], dim=-1) # [NF, L]
            # if (torch.log2(neg_prediction) <= -1e12).any():
            #     breakpoint()

            v_information -= torch.log2(neg_prediction)
        
        pbar.set_description(f'number of inf: {(v_information > 1e20).sum()}/{v_information.numel()}')

        # print(data_idx)
        if data_idx % save_every == 0:
            torch.save(v_information, v_information_save_dir)

    v_information /= training_dataset.num_samples


if __name__ == '__main__':
    # d = FeatureDataset(torch.LongTensor().unsqueeze(-1).expand(3, 2, 10), torch.LongTensor())
    # for each in d:
    #     print(each)
    # lprof = line_profiler.LineProfiler()
    # lprof.add_function(mlp)
    # os.environ['LINE_PROFILE'] = '1'
    CLI([mlp, independent_mlp, truncate_neg_samples, v_information], )
    # lprof.print_stats()
    # fd = FeatureDataset(
    #     [torch.randn(10, 128), torch.randn(3, 128), torch.randn(8, 128)],
    #     [torch.randint(0, 2, (10,)), torch.randint(0, 2, (3,)), torch.randint(0, 2, (8,))],
    #     batch_size=4,
    # )
    # for inputs, labels in tqdm(fd, desc='iterating...', total=len(fd) - 1):
    # for inputs, labels in fd:
    #     print(inputs.shape, labels.shape)
    