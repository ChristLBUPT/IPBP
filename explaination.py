import scipy
from matplotlib import pyplot as plt
from scipy.integrate import quad_vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import tensor, Tensor
from scipy.integrate import quad, dblquad, quad_vec
from scipy.stats import spearmanr
from sklearn.neighbors import KernelDensity
import torch
from torch import Tensor, tensor, nn
import numpy as np
from tqdm import tqdm, trange
from typing import List, Dict, Union
import math
import pdb
from dep_model import InputExample
import os.path as osp


class FeatureList:
    def __init__(self):
        self.item = None
    def append(self, new_item: Tensor):
        if self.item is None:
            self.item = new_item
        else:
            self.item = torch.cat([self.item, new_item], dim=0)


def adj_matrix_to_heights(adj_matrix: Tensor) -> Tensor:
    """
    Args:
        adj_matrix: adjacency matrix for the graph with edges pointing from child to parent
    """
    adj_matrix = adj_matrix.T
    in_degrees = adj_matrix.sum(dim=0)
    out_degrees = adj_matrix.sum(dim=1)
    root_idx = ((in_degrees == 0) & (out_degrees != 0)).nonzero().item()
    # print(f'root is {root_idx}')
    # start bfs
    nnodes = adj_matrix.size(0)
    node_queue = [0] * (nnodes + 10)
    heights = [0] * nnodes
    cptr = 0 
    rear_ptr = 1
    node_queue[0] = root_idx
    heights[root_idx] = 0
    while cptr < nnodes:
        for child_node_idx, has_edge in enumerate(adj_matrix[node_queue[cptr]]):
            if has_edge:
                node_queue[rear_ptr] = child_node_idx
                rear_ptr += 1
                heights[child_node_idx] = heights[node_queue[cptr]] + 1
        cptr += 1
    
    # print(heights)
    return adj_matrix.T * torch.tensor(heights, device=adj_matrix.device).unsqueeze(-1).expand_as(adj_matrix)

def get_shape(array_like_item):
    shape = []
    while hasattr(array_like_item, '__len__'):
        if isinstance(array_like_item, torch.Tensor):
            shape.append(f"{[*array_like_item.shape]}(T)")
            break
        shape.append(f"{len(array_like_item)}(A)") 
        array_like_item = array_like_item[0]
    
    return f"[{', '.join(shape)}]"

def print_listlike(listlike, chunk: int = 10):
    for start in range(0, len(listlike), chunk):
        print(*listlike[start: start + chunk])

def move_to_device(device: Union[int, str], *tensors):
    return [each.to(device) if each is not None else each for each in tensors]


def visualize_attn_features(arc_features, label_features, labelmap: Dict[str, int], algorithm: bool = ['pca', 'tsne']):
    labelmap_inv = {label_name: label_idx for label_idx, label_name in labelmap.items()}
    pos_features, neg_features = arc_features
    dimreduce = PCA(2) if algorithm == 'pca' else TSNE(2)
    # concatenate along the first dim, pos_features become [samp1_num_arcs + samp2_num_arcs ... + sampN_num_arcs, num_layers * num_heads]
    pos_features = torch.cat(pos_features)
    neg_features = torch.cat(neg_features)
    print(f"applying {algorithm} transform on all examples...")
    attn_features_dim2 = dimreduce.fit_transform(torch.cat([pos_features, neg_features]).float())#.cpu().view(-1, attn_last_feature_dim))#.reshape(S, S, -1)
        # print(pos_samples.shape, neg_samples.shape)
    fig = plt.figure(figsize=(10, 10))
    # print(neg_samples.T.shape, (torch.ones(neg_samples.shape[0]) * 0.01).shape)
    pos_features_dim2 = attn_features_dim2[: pos_features.shape[0]] # [total_num_arcs, 2]
    neg_features_dim2 = attn_features_dim2[pos_features.shape[0]: ] # [total_num_nonarcs(\sum_i=1^N(seq_len_i * seq_len_i) - total_num_arcs), 2]

    plt.scatter(*neg_features_dim2.T, s=0.4, c='green')
    plt.scatter(*pos_features_dim2.T, s=0.8, c='red')

    dimreduce_for_labeled = PCA(2) if algorithm == 'pca' else TSNE(2)
    label_features_list = []
    label_features_indices = []
    for label_idx, this_label_features in label_features.items():
        label_features_list.append(torch.cat(this_label_features))
        label_features_indices.append(label_idx)
    
    plt.show()

    labeled_features_dim2 = dimreduce_for_labeled.fit_transform(torch.cat(label_features_list).float())
    fig = plt.figure(figsize=(10, 10))
    lb, rb = 0, label_features_list[0].shape[0]
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X']
    for label_idx in range(len(label_features_list)):
        lb = rb
        rb += label_features_list[label_idx].shape[0]
        label_name = labelmap_inv[label_features_indices[label_idx]]
        this_label_features_dim2 = labeled_features_dim2[lb: rb]
        print(f'{label_name = }, decomposed_shape = {this_label_features_dim2.shape}')
        plt.scatter(*this_label_features_dim2.T, s=2, marker=markers[label_idx % len(markers)], label=label_name)
        # plt.()
    
    # plt.legend()
    plt.show()

def concatenate_attn_features(arc_features, height_features, label_features, concat_across_labels: bool = False):
    """
    Returns:
        arc_features_concat: Tensor of [total_num_arcs, feature_dim(n_layers * n_heads)]
        arc_cumsums: 
    """
    pos_features, neg_features = torch.cat(arc_features[0]), (torch.cat(arc_features[1]) if len(arc_features[1]) > 0 else torch.empty([0, arc_features[0][0].shape[-1]]))
    n_pos_features, n_neg_features = pos_features.shape[0], neg_features.shape[0]
    arc_features_concat = torch.cat([pos_features, neg_features]) if concat_across_labels else (pos_features, neg_features)
    height_features_concat = None
    height_num_attn_vectors = []
    
    label_features_concat = None
    label_num_attn_vectors = [] # number of attn vectors of each label
    label_names = []
    # concatenate height features of each height
    for idx, (height, height_feature) in enumerate(height_features.items()):
        this_height_features_concat = torch.cat(height_feature)
        if height_features_concat is None:
            height_features_concat = this_height_features_concat if concat_across_labels else (this_height_features_concat, )
        else:
            height_features_concat = torch.cat([height_features_concat, this_height_features_concat]) if concat_across_labels else height_features_concat + (this_height_features_concat, )
        height_num_attn_vectors.append(this_height_features_concat.shape[0])

    # concatenate label features of each label
    for idx, (label_name, label_feature) in enumerate(label_features.items()):
        this_label_features_concat = torch.cat(label_feature)
        if label_features_concat is None:
            label_features_concat = this_label_features_concat if concat_across_labels else (this_label_features_concat, )
        else:
            label_features_concat = torch.cat([label_features_concat, this_label_features_concat]) if concat_across_labels else label_features_concat + (this_label_features_concat, )
        label_names.append(label_name)
        label_num_attn_vectors.append(this_label_features_concat.shape[0])

    if concat_across_labels:
        return arc_features_concat, np.cumsum([0, n_pos_features, n_neg_features]).tolist(), \
            height_features_concat, np.cumsum(height_num_attn_vectors).tolist(), \
            label_features_concat, np.cumsum(label_num_attn_vectors).tolist(), label_names 
    else:
        return arc_features_concat, height_features_concat, label_features_concat, label_names

def zip_features_with_y(features: Tensor, cumsums: List[str]):
    features = features.T # Tensor[num_features, total_num_samples]
    all_zipped = []
    for feature_idx in trange(features.shape[0], desc=f'zipping (total {len(cumsums) - 1} num_rel_categories)'):
        this_feature_samples = features[feature_idx] # Tensor[total_num_samples]
        ys = torch.cat([torch.ones(cumsums[y + 1] - cumsums[y]) * y for y in range(len(cumsums) - 1)])
        all_zipped.append(torch.stack([this_feature_samples, ys]))
    
    return all_zipped

def integral_torch_cuda(x: Tensor, y: Tensor, definite: bool = True):
    """
    use trapzoid method to estimate y's integral on the interval defined by x
    """
    x_diffs = x[1:] - x[: -1] # heights of trapzoids
    y_sums = y[:-1] + y[1:] # sum of (upper and lower) bases
    trapzoid_areas = (x_diffs) * (y_sums) / 2 #
    return trapzoid_areas.sum() if definite else torch.cat([torch.Tensor([0]).to(x.device), trapzoid_areas.cumsum(-1)])

# def inference_by_func(values: Tensor, x: Tensor, y: Tensor ):
#     """
#     Definition: y = f(x), estimate f(values)
#     Args:
#         values: the values to lerp
#         x: the domain of defination
#         y: the function values over x
#     """
#     # x = torch.cat([torch.arange(-1.2, 0, 0.1), torch.arange(0, 1, 0.05), torch.arange(1, 2.2, 0.1)])
#     original_value_shape = values.shape
#     x = x.view(-1)
#     # pdb.set_trace()
#     values = values.view(-1, 1)
#     out_of_bounds_mask = (values < x.min()) | (values > x.max())
#     # values.clamp_(x.min() + 1e-8, x.max() - 1e-8)
#     values[out_of_bounds_mask] = ((x.min() + x.max()) / 2).to(values.dtype)
#     diff = values - x.unsqueeze(0)
#     is_pos_diff = diff >= 0
#     indices = is_pos_diff.sum(-1) # the index of each first point that is greater than the value
#     sample_dist_l = torch.gather(diff, -1, (indices - 1).unsqueeze(-1)).view(-1)
#     sample_dist_r = torch.gather(diff, -1, indices.unsqueeze(-1)).view(-1).abs()
#     # for sample_idx, (values_val, index) in enumerate(zip(values.view(-1).tolist(), indices.view(-1).tolist())):
#     #     print(f'value {values_val:.3f} between {x[index - 1]:.3f} and {x[index]:.3f}, weights: {sample_dist_l[sample_idx]:.3f}, {sample_dist_r[sample_idx]:.3f}')    
#     ys = y[indices - 1]
#     ys_next = y[indices]
#     lerp_weight_l = sample_dist_r / (sample_dist_l + sample_dist_r)
#     lerp_weight_r = sample_dist_l / (sample_dist_l + sample_dist_r)

#     func_values = lerp_weight_l * ys + lerp_weight_r * ys_next
#     func_values[out_of_bounds_mask.view(*func_values.shape)] = 0

#     return func_values.view(original_value_shape)

def inference_by_func(values, x, y):
    """
    Definition: y = f(x), estimate f(values)
    Args:
        valuesues: the valuesues to lerp
        x: the domain of defination
        y: the function valuesues over x
    """
    # Find the indices of the bracketing points in x, and clamp them within the range of x
    indices = torch.searchsorted(x, values)
    indices = torch.clamp(indices, 1, len(x) - 1) # 
    oob_mask = (values < x.min()) | (values > x.max())

    x0 = x[indices - 1] # Get the bracketing x and y valuesues
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]
    slope = (y1 - y0) / (x1 - x0)
    y_interp = y0 + slope * (values - x0)
    y_interp[oob_mask] = 0
    return y_interp


def batched_inference_by_func(values, x, y):
    """
    Definition: y = f(x), estimate f(values) for batched inputs.
    Args:
        values: the values to lerp in shape [batch_size, n_values]
        x: the domain of definition in shape [batch_size, n_x]
        y: the function values over x in shape [batch_size, n_x]
    """
    n_x = x.shape[-1]
    n_values = values.shape[-1]

    # Find the indices of the bracketing points in x for each batch and clamp them within the range of x
    indices = torch.searchsorted(x, values, right=True)
    indices = torch.clamp(indices, 1, n_x - 1)

    # Create an out-of-bounds mask for each batch
    oob_mask = (values < x.min(dim=-1, keepdim=True).values) | (values > x.max(dim=-1, keepdim=True).values)

    # Gather x0, x1, y0, and y1 based on the computed indices for each batch
    x0 = torch.gather(x, -1, indices - 1)
    x1 = torch.gather(x, -1, indices)
    y0 = torch.gather(y, -1, indices - 1)
    y1 = torch.gather(y, -1, indices)

    # Calculate the slope and interpolate y values
    slope = (y1 - y0) / (x1 - x0)
    y_interp = y0 + slope * (values - x0)

    # Set interpolated values to zero where they are out of bounds
    y_interp[oob_mask] = 0
    return y_interp

@torch.no_grad()
def estimate_kde_torch(
    x: torch.Tensor, 
    data_points: torch.Tensor, 
    weights: torch.Tensor = None, 
    normalize: bool = True, 
    bw: float = None, 
    debug: bool = False
) -> Tensor: 
    if debug:
        print(f'{x.numel():,}, {data_points.numel():,}')
    assert x.device == data_points.device, f"error, 'x' and 'data_points' in different device: '{x.device}' and '{data_points.device}'"
    device = x.device
    n_x = x.shape[0]
    original_x = x.clone()
    n_samples = data_points.shape[0]
    n_dims = 1
    if weights is None:
        weights = torch.ones(n_samples, device=device) / n_samples # default
    else:
        weights = weights.view(n_samples)
        weights = weights / weights.sum()
    if bw is None:
        neff = 1 / (weights ** 2).sum()
        bw = (neff ** (-1. / (n_dims + 4))).item()
    
    x = x.unsqueeze(-1).expand(n_x, n_samples)
    data_points = data_points.unsqueeze(0).expand(n_x, n_samples)
    diff = (x - data_points).abs() / bw
    norm = (math.pow((2 * torch.pi), (- n_dims / 2.)) / torch.sqrt(torch.std(data_points))).item()
    kernel = torch.exp(- diff ** 2) * norm
    raw_pdf = kernel.mean(-1) / bw
    if normalize:
        cdf_max = integral_torch_cuda(original_x, raw_pdf)
        # print(bw, cdf_max, 1 / cdf_max)

    return raw_pdf / cdf_max if normalize else raw_pdf

def estimate_mi(x: Tensor, label_probabilities: List[float], marginal_probabilities: List[Tensor], joint_probabilities: List[List[Tensor]], device: str = 'cuda:0', visualize: bool = False, intermediate_save_dir: str = None):
    """
    Args:
        x: (Tensor[n_x]) domain of probab distributions
        label_probabilities: (List[n_labels] of float) probabilities of each discrete (label) variable (p(y=0), p(y=1), p(l=0), p(l=1), ... p(l=m))
        marginal_probabilities: (List[n_heads] of Tensor[n_x]) PDF of each continuous variable (attention head): f(a_1), f(a_2), ... f(a_h)
        joint_probabilities(List[n_heads, n_labels]): joint probability of each feature, each label \\
            f(a_1, l=0), f(a_1, l=1), ..., f(a_1, l=m) \\
                            ... \\
            f(a_h, l=0), f(a_h, l=1), ..., f(a_h, l=m) 
    """
    def plot_mi_component(ax: plt.Axes, x: Tensor, values: Tensor, title: str):
        ax.plot(x.detach().cpu().numpy(), values.detach().cpu().numpy())
        ax.set_title(title)

    mi_list = []
    eps = 1e-8
    if intermediate_save_dir is not None:
        excess_surprises, weighted_excess_surprises = [], []
    for feature_idx in range(len(joint_probabilities)):#, desc=f'estimating MI ({len(label_probabilities)} discrete classes)'):
        if intermediate_save_dir is not None:
            excess_surprises.append([])
            weighted_excess_surprises.append([])
        mi = 0
        for m in range(len(joint_probabilities[feature_idx])): # m = 0(neg) or 1(pos)
            this_joint_probability = joint_probabilities[feature_idx][m].float().to(device) # f(a_i, y=y_m(1 or 0))
            # print(len(marginal_probabilities), len(joint_probabilities), feature_idx)
            this_marginal_probab_attn = marginal_probabilities[feature_idx].float().to(device)  # f(a_i)
            information_content = torch.log2(this_joint_probability) - torch.log2(this_marginal_probab_attn) - torch.log2(torch.tensor(label_probabilities[m]))
            information_content.masked_fill_(this_joint_probability == 0, 0)
            # print(information_content)
            mi += integral_torch_cuda(
                x.to(device), 
                (this_joint_probability * information_content).to(device)
            )
            if visualize:
                fig = plt.figure(figsize=(8, 8))
                axes = fig.subplots(4, 1)
                plot_mi_component(axes[0], x, this_joint_probability, f"joint: f(a_{feature_idx}, t{m})")
                plot_mi_component(axes[1], x, this_marginal_probab_attn, f"marginal: f(a_{feature_idx})")
                plot_mi_component(axes[2], x, information_content, f"excess surprise: log[f(a_{feature_idx}, t{m})/f(a_{feature_idx}) * p(t)]")
                plot_mi_component(axes[3], x, this_joint_probability * information_content, f"weighted excess surprise: f(a_{feature_idx}, t{m}) * log[f(a_{feature_idx}, t{m})/f(a_{feature_idx}) * p(t)]")
                fig.tight_layout()
                fig.show()
                raise KeyboardInterrupt
            if intermediate_save_dir is not None:
                excess_surprises[-1].append(information_content)
                weighted_excess_surprises[-1].append(this_joint_probability * information_content)
            if mi.item() == torch.inf:
                breakpoint()
                print(mi.item())
        mi_list.append(mi.item())
    
    if intermediate_save_dir is not None:
        torch.save(excess_surprises, osp.join(intermediate_save_dir, 'excess_surprises.pt'))
        torch.save(weighted_excess_surprises, osp.join(intermediate_save_dir, 'weighted_excess_surprises.pt'))
        torch.save(joint_probabilities, osp.join(intermediate_save_dir, 'joint_probabilities.pt'))
        torch.save(marginal_probabilities, osp.join(intermediate_save_dir, 'marginal_probabilities.pt'))
    
    return mi_list

"""
>>>def infer_by_attn_features_with_labels(
>>>    conditional_probabs: List[Tuple[Tensor, Tensor]], 
>>>    mutual_informations: List[float],
>>>    label_conditional_probabs: List[List[Tensor]],
>>>    label_mutual_informations: List[List[float]],
>>>    tok: LlamaTokenizer, model: LlamaModel, dataloader: DataLoader, labelmap: Dict[str, int], num_samples: int = 100, silent: bool = True, 
>>>    mi_threshold: float = 0.01,  
>>>    use_weighted_mi: bool = True
>>>) -> Tuple[
>>>        Tuple[List[Tensor], List[Tensor]], Dict[str, List[Tensor]]
>>>    ]:
>>>    
>>>    Infers the dependency tree and labels (arc types) using attention feature values
>>>    and the corresponding probability distributions.
>>>    Steps:
>>>    - First calculate label scores (arc labels like `nsubj`, `obj`)
>>>    - Then infer the dependency arcs based on label scores.
>>>    - Filter attention features based on their mutual information (MI) values.
>>>    
>>>    tok.pad_token = tok.eos_token
>>>    torch.backends.cuda.enable_flash_sdp(True)
>>>    torch.backends.cuda.enable_mem_efficient_sdp(False)
>>>
>>>    if num_samples == -1:
>>>        num_samples = len(dataloader)
>>>
>>>    n_correct_arcs, n_correct_labels, total_arcs = 0, 0, 0
>>>    
>>>    for data_idx, (input_ids, input_attention_mask, label_mask, eval_mask, \
>>>        arcs, rels, word_ids, pos_ids, ngram_ids, \
>>>        ngram_positions, segment_ids, valid_ids) in enumerate(pbar := tqdm(dataloader, total=num_samples, desc='extracting attentions')):
>>>
>>>        echo = not silent and data_idx == 0
>>>
>>>        if echo:
>>>            print(input_ids.shape, label_mask.shape, valid_ids.shape, arcs.shape, rels.shape)
>>>        
>>>        B, S = input_ids.shape
>>>        w2s = []  # Whole-word to subword map, only keep last subword position for each word
>>>        for subword_idx, each in enumerate(valid_ids[0].tolist()):
>>>            if each == 1:
>>>                w2s.append(subword_idx)
>>>
>>>        arc_adj_matrix = torch.zeros(S, S).to(MODEL_DEVICE)  # 1 for `have arc`, 0 for `no arc`
>>>        label_adj_matrix = torch.zeros(S, S).to(MODEL_DEVICE)  # Store label arc types
>>>
>>>        for word_idx, head_idx in enumerate(arcs[0]):
>>>            if head_idx != -1:
>>>                arc_adj_matrix[w2s[word_idx]][w2s[head_idx]] = 1
>>>                label_adj_matrix[w2s[word_idx]][w2s[head_idx]] = rels[0][word_idx]
>>>
>>>        # Forward pass to get attention scores
>>>        res = model.forward(
>>>            input_ids=input_ids, attention_mask=input_attention_mask,
>>>            output_hidden_states=True,
>>>            output_attention_queries=True
>>>        )
>>>        key_values, queries = res.past_key_values, res.queries
>>>
>>>        attn_scores = ()  # [num_layers, 1(batch_size), num_heads, seq_len, seq_len]
>>>        for layer_idx in range(len(model.layers)):
>>>            k = key_values[layer_idx][0]
>>>            q = queries[layer_idx]
>>>            n_groups = q.shape[-3] // k.shape[-3] if q.shape[-3] != k.shape[-3] else 1
>>>            k = torch.repeat_interleave(k, n_groups, -3)
>>>            this_attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(res.last_hidden_state.shape[-1])
>>>            attn_scores += (this_attn_score,)
>>>        
>>>        attn_scores = torch.cat(attn_scores, dim=1).squeeze(0)
>>>
>>>        # Step 1: Compute Label Scores (Dependency Label Inference)
>>>        probab_labels = torch.zeros(S, S, len(labelmap)).to(MODEL_DEVICE)  # One-hot for each arc type (relation label)
>>>        
>>>        for feature_idx in range(attn_scores.shape[0]):
>>>            # Apply MI-based filtering
>>>            for label_idx, label_probab in enumerate(label_conditional_probabs[feature_idx]):
>>>                if label_mutual_informations[label_idx][feature_idx] > mi_threshold:
>>>                    weight = mutual_informations[feature_idx] if use_weighted_mi else 1.0  # Weight by MI if specified
>>>                    probab_labels[:, :, label_idx] += weight * torch.log2(
>>>                        inference_by_func(
>>>                            attn_scores[feature_idx].flatten().to(MODEL_DEVICE), x.to(MODEL_DEVICE), label_probab.flatten().to(MODEL_DEVICE)
>>>                        ).view(S, S)
>>>                    )
>>>
>>>        # Normalize label probabilities to make valid scores
>>>        probab_labels = torch.softmax(probab_labels, dim=-1)
>>>
>>>        # Step 2: Infer arcs based on highest label scores
>>>        # We'll use the label scores to guide the arc inference.
>>>        predicted_labels = torch.argmax(probab_labels, dim=-1)  # Get most likely label for each arc
>>>
>>>        # Step 3: Aggregate scores for arc presence using label information
>>>
>>>        probab_arcs = probab_labels.max(dim=-1)[0]  # Use the max label score as the arc score
>>>        label_mask[:, 0] = 0  # Mask first token (often <s> token)
>>>
>>>        # Infer arcs using Eisner’s algorithm, using the label-driven arc probabilities
>>>        predicted_arcs = eisner(probab_arcs[valid_ids[0] == 1][:, valid_ids[0] == 1].unsqueeze(0), label_mask)
>>>
>>>        # pdb.set_trace()
>>>        # Calculate UAS and LAS
>>>        correct_arc_mask = (predicted_arcs == arcs)[eval_mask]
>>>        # correct_label_mask = (predicted_labels == rels)[eval_mask]
>>>
>>>        n_correct_arcs += correct_arc_mask.sum().item()
>>>        # n_correct_labels += (correct_arc_mask & correct_label_mask).sum().item()
>>>        total_arcs += correct_arc_mask.shape[0]
>>>
>>>        pbar.set_postfix({
>>>            "UAS": round(n_correct_arcs / total_arcs * 100, 2),
>>>            # "LAS": round(n_correct_labels / total_arcs * 100, 2)
>>>        })
>>>
>>>        if data_idx >= num_samples - 1:
>>>            break
>>>
>>>    return n_correct_arcs / total_arcs, n_correct_labels / total_arcs
"""

def test_lerp(values, xs, ys):
    func_values = inference_by_func(values, xs, ys)
    plt.figure()
    plt.plot(xs, ys, 'ro--', )
    plt.scatter(values, func_values)
# test_lerp(torch.rand(8) * 1.98 - 1.98 / 2, torch.arange(-1, 1.0001, 0.1), torch.randn(21) * 5 - 1)
def find_example_with_maximum_label_variety(train_data: List[InputExample]):
    max_index = -1
    max_num_rels = -1
    for idx, each in enumerate(train_data):
        num_rels = len(set(each.label))
        if num_rels > max_num_rels:
            max_num_rels = num_rels
            max_index = idx
    
    return max_index

def formalize_dependency_output(input_example: InputExample):
    """
    Args:
        input_example: the input example consisting of the following fields:
            text_a: List[str], the original sentence
            head: the head token idx of each token
            label: the dependency label of each token
    Returns:
        result: a string that formulates in the "<token> -> <head> (label)" manner for each token in `text`
    """
    text, head, label = input_example.text_a, input_example.head, input_example.label
    result = ''
    for idx, token in enumerate(text[1:]):
        result += f'{token} -> {text[head[idx]]} ({label[idx]}), '
    return result

# print(train_data_llama[max_index].text_a)
# # print(train_data_llama[max_index].head)
# print(formalize_dependency_output(['root'] + train_data_llama[max_index].text_a.split(' '), train_data_llama[max_index].head, train_data_llama[max_index].label))
    

# torch.cuda.set_device(0)
# os.environ['HTTP_PROXY'] = 'http://172.18.214.116:6666'
# os.environ['HTTPS_PROXY'] = 'http://172.18.214.116:6666'
# load model
# ELM_PATH = '../pretrained-models/OpenELM-3B/'
# MODEL_PATH = '../pretrained-models/opt-1.3b/'
# print(f'initializing ELM tokenizer and model...')
# dep_parser_elm = DependencyParser(
#     json.load(open('./configs/elm/sample_labelmap.json')), 
#     Hparam(**json.load(open('./configs/elm/sample_hpara.json'))), 
#     ELM_PATH
# )
# model_elm, tok_elm = dep_parser_elm.llm, dep_parser_elm.tokenizer
# model_elm = model_elm.to('cuda:0').to(torch.bfloat16)

# for _ in trange(10_000):
#     adj_matrix_to_heights(torch.Tensor([
#         [0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0]
#     ]))

##############################################################################################################################
#                                                                                                                            #
#           (DEPRECATED): concatenate attention features across the different groups / labels / heights                      #
#                                                                                                                            #
##############################################################################################################################
# print(arc_attn_features.shape, arc_cumsum, label_attn_features.shape)
# arc_attn_features, arc_cumsum, height_attn_features, height_cumsum, label_attn_features, label_cumsum, label_names = \
#     concatenate_attn_features(arc_attn_features, height_attn_features, label_attn_features)

# arc_feature_with_y = zip_features_with_y(arc_attn_features, arc_cumsum)
# height_features_with_y = zip_features_with_y(height_attn_features, height_cumsum)
# label_feature_with_y = zip_features_with_y(label_attn_features, label_cumsum)

# 
# with open(osp.join(kde_save_pth, 'labelmap.json'), 'w') as f:
#     json.dump(dep_parser_llama.labelmap, f)
################################################################################
# print(dep_parser_llama.labelmap)
# print(len(label_features))
# print(label_id2name)
# print(label_names)
################################################################################
# labelmap = dep_parser_llama.labelmap
# label_id2name = {label_id: label_name for label_name, label_id in dep_parser_llama.labelmap.items()}
    
# print(arc_entropy, label_entropy)
# print()
# attn_entropy = [
#     integral_torch_cuda(
#         x.float().to(KDE_DEVICE), 
#         (-this_marginal_probab_attn.float() * torch.log2(this_marginal_probab_attn).float()).masked_fill_(this_marginal_probab_attn == 0, 0).to(KDE_DEVICE)
#     ) for this_marginal_probab_attn in marginal_probab_attn
# ]
# print(marginal_probab_attn)
# print([(-this_marginal_probab_attn.float() * torch.log2(this_marginal_probab_attn).float()) for this_marginal_probab_attn in marginal_probab_attn])
# print(label_entropy)
# print(attn_entropy)

# normal output:
# [7]possessive MI(0.12390271574258804) greater than entropy(0.07756942592831961) at head 25
# [19]root MI(1.3522238731384277) greater than entropy(0.24868394869673321) at head 297
# [44]csubjpass MI(0.000554161611944437) greater than entropy(0.0003533460351081003) at head 2
# 0.2675654170487565 0.4516241210517546
# 0.5388150031295511

# transpose output:
# [19]root MI(1.1840410232543945) greater than entropy(0.24868394869673321) at head 285
# [39]prt MI(0.028513232246041298) greater than entropy(0.027746423344412994) at head 736
# [44]csubjpass MI(0.0005773947341367602) greater than entropy(0.0003533460351081003) at head 14
# 0.2587862276010542 0.40234241863714487
# 0.5291863643721905

# ========================================================================================================================
#      (pasted from notebook) get all attention samples of heads with binary-MI proportions more than 1
# ========================================================================================================================
# current_masked_idx = 0
# masked_idx_to_unmasked_idx = {}
# for idx in range(len(label_features)):
#     if idx not in masked_label_indices:
#         masked_idx_to_unmasked_idx[current_masked_idx] = idx
#         current_masked_idx += 1
# # print(labelmap['root'])
# # print(masked_idx_to_unmasked_idx[19])
# # print(label_features[20].shape)
# print(label_features[masked_idx_to_unmasked_idx[19]][:, 285])
# print(label_features[masked_idx_to_unmasked_idx[39]][:, 736])
# print(label_features[masked_idx_to_unmasked_idx[44]][:, 14])
# # len(label_features)
# ========================================================================================================================
#      (pasted from notebook) get (arc and label) MI proportions (MI/discrete variable entropy) of each head and list most informative heads
# ========================================================================================================================
# mi_proportions_arc = [*enumerate([each / arc_entropy for each in arc_mi])]
# mi_proportions_label = [*enumerate([each / label_entropy for each in label_mi])]
# # print(mi_proportions)
# print(sorted(mi_proportions_arc, reverse=True, key=lambda x: x[1]))
# print(sorted(mi_proportions_label, reverse=True, key=lambda x: x[1]))
if __name__ == "__main__":
    print(values := torch.arange(9).view(3, 3).float())
    print(inference_by_func(values, torch.arange(-10, 20).float(), torch.arange(-9, 21).float()))