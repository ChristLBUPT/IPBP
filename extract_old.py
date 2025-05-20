def extract_attn_features(
    tok: LlamaTokenizer, 
    model: LlamaModel, 
    dataloader: DataLoader, 
    labelmap: Dict[str, int], 
    num_samples: int = 100, 
    silent: bool = True, 
    transpose: bool = False, 
    save: bool = False,
    save_every: int = -1,
) -> Tuple[
        Tuple[Union[List[Tensor], FeatureList], Union[List[Tensor], FeatureList]], Dict[str, Union[List[Tensor], FeatureList]]
    ]:
    """
    returns 
        (
            pos_features: List[Tensor[n_arcs_of_sample_i, feature_dim(n_layers * n_heads)]], 
            neg_features: List[Tensor[sample_i_seq_len ^ 2 - n_arcs_of_sample_i, feature_dim]]
        ), label_features: Dict[str(label_name of rel j), List[Tensor[samp_i_n_rel_j_arcs, feature_dim]]]
    """
    tok.pad_token = tok.eos_token
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # if concat:
    #     pos_features, neg_features = FeatureList(), FeatureList()
    #     height_features, label_features = defaultdict(FeatureList), defaultdict(FeatureList)
    # else:
    pos_features, neg_features = [], []
    height_features = defaultdict(list)
    label_features = defaultdict(list)
    if num_samples == -1:
        num_samples = len(dataloader)
    if save:
        save_idx = 0
        num_saves = math.ceil(num_samples / save_every)
        n_pos_samples, n_neg_samples = 0, 0
    
    id2label = {label_idx: label_name for label_name, label_idx in labelmap.items()}
    for data_idx, (input_ids, input_attention_mask, label_mask, eval_mask, \
        arcs, rels, word_ids, pos_ids, ngram_ids, \
        ngram_positions, segment_ids, valid_ids) in enumerate(tqdm(dataloader, total=num_samples, desc='extracting attentions', ncols=100)):

        echo = not silent and data_idx == 0

        if echo:
            print(input_ids.shape, label_mask.shape, valid_ids.shape, arcs.shape, rels.shape)
        B, S = input_ids.shape
        w2s = [] # the idx at position i is whole-word i's last subword's idx
        for subword_idx, each in enumerate(valid_ids[0].tolist()):
            if each == 1:
                w2s.append(subword_idx)

        arc_adj_matrix = torch.zeros(S, S).to('cuda:0') # 1 for `have arc`, 0 for `no arc` (marking at whole-word's last subword)
        label_adj_matrix = torch.zeros(S, S).to('cuda:0') # arc[i][j]'s relation type (marking at whole-word's last subword)
        # print_listlike([*enumerate(tok.convert_ids_to_tokens(input_ids[0]))])
        for word_idx, head_idx in enumerate(arcs[0]):
            if head_idx != -1:
                arc_adj_matrix[w2s[word_idx]][w2s[head_idx]] = 1
                label_adj_matrix[w2s[word_idx]][w2s[head_idx]] = rels[0][word_idx]
                # print(input_ids[0][w2s[word_idx]], input_ids[0][w2s[head_idx]])
                dependant_pos_id, head_pos_id = w2s[word_idx], w2s[head_idx]
                dependant_token_id, head_token_id = input_ids[0][dependant_pos_id].item(), input_ids[0][head_pos_id].item()
                # print(f"[{dependant_pos_id}]{tok.convert_ids_to_tokens(dependant_token_id)} -> [{head_pos_id}]{tok.convert_ids_to_tokens(head_token_id)} ({id2label[rels[0][word_idx].item()]})")
                # print([*enumerate(tok_llama.convert_ids_to_tokens(input_ids[0]))])

        height_adj_matrix = adj_matrix_to_heights(arc_adj_matrix)
        if transpose:
            arc_adj_matrix, height_adj_matrix, label_adj_matrix = arc_adj_matrix.T, height_adj_matrix.T, label_adj_matrix.T
        res = model.forward(
            input_ids=input_ids, attention_mask=input_attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            output_attention_queries=True
        )
        key_values, queries = res.past_key_values, res.queries
        # kv: [num_layers, 2(k and v), batch_size, num_heads, sequence_length, head_dim], q: [num_layers, batch_size, num_heads, sequence_length, head_dim]
        if echo:
            print('past_key_values shape:', get_shape(res.past_key_values))
            # print(key_values)
            print('queries shape:', get_shape(res.queries))
            # print(queries)

        attn_scores = () # [num_layers, 1(batch_size), num_heads, seq_len, seq_len]
        for layer_idx in range(len(model.layers)):
            k = key_values[layer_idx][0]
            q = queries[layer_idx]
            if q.shape[-3] != k.shape[-3]: # num_q_heads == num_k_heads * n_groups
                n_groups = q.shape[-3] / k.shape[-3]
                assert n_groups == int(n_groups)
                n_groups = int(n_groups)
            else:
                n_groups = 1
            
            k = torch.repeat_interleave(k, n_groups, -3)
            this_attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(res.last_hidden_state.shape[-1]) 
            # scaled dot-product attention, this_attn_score: [batch_size(1), num_heads, sequence_length, sequence_length]
            attn_scores += (this_attn_score,)
        
        if echo:
            print('attn_scores shape:', get_shape(attn_scores))
        attn_scores = torch.cat(attn_scores, dim=1).squeeze(0) # [num_heads * num_layers, sequence_length, sequence_length], batch_size(1) is squeezed
        if echo:
            print('attn_score after squeezing:', get_shape(attn_scores))
        attn_features = attn_scores.permute(1, 2, 0) # [sequence_length, sequence_length, num_heads * num_layers]
        # add flatten attention features
        pos_features.append(attn_features[(arc_adj_matrix == 1).cpu()].cpu()) 
        # [[sample1_num_arcs, num_heads * num_layers], [sample2_num_arcs, num_heads * num_layers], ... ]
        neg_features.append(attn_features[(arc_adj_matrix == 0).cpu()].cpu()) 
        # [[samp1_seq_len * samp1_seq_len - sample1_num_arcs, num_heads * num_layers], [samp2_seq_len * samp2_seq_len - sample2_num_arcs, num_heads * num_layers]]
        for label_idx in labelmap.values():
            label_features[label_idx].append(attn_features[(label_adj_matrix == label_idx).cpu()].cpu())
            # {rel1_label_name: [[samp1_num_rel1_samples, num_heads * num_layers]], [[samp2_num_rel1_samples, num_heads * num_layers]]}
        
        for height in range(1, int(height_adj_matrix.max().item() + 1)):
            height_features[height].append(attn_features[(height_adj_matrix == height).cpu()].cpu())

        # save during
        if save and data_idx != 0 and data_idx % save_every == 0 or data_idx == num_samples - 1:
            print(f'saving attention samples at {data_idx} (save idx {save_idx})...')
            (pos_features_concat, neg_features_concat), height_features_concat, label_features_concat, label_ids = concatenate_attn_features(
                (pos_features, neg_features), height_features, label_features, concat_across_labels=False)

            save_attn_samples(pos_features_concat, osp.join(kde_save_pth, formalize_pickle_file_name(f'pos_attn_samples.pkl', save_idx, num_saves)))
            save_attn_samples(neg_features_concat, osp.join(kde_save_pth, formalize_pickle_file_name('neg_attn_samples.pkl', save_idx, num_saves)))
            save_attn_samples(height_features_concat, osp.join(kde_save_pth, formalize_pickle_file_name('height_attn_samples.pkl', save_idx, num_saves)))
            save_attn_samples(label_features_concat, osp.join(kde_save_pth, formalize_pickle_file_name('label_attn_samples.pkl', save_idx, num_saves)))
            n_pos_samples += len(pos_features_concat)
            n_neg_samples += len(neg_features_concat)
            pos_features, neg_features = [], []
            height_features = defaultdict(list)
            label_features = defaultdict(list)
            save_idx += 1

        if data_idx >= num_samples - 1:
            break

    if save:
        with open(osp.join(kde_save_pth, 'labelmap.json'), 'w') as f:
            json.dump(labelmap, f)
        with open(osp.join(kde_save_pth, formalize_pickle_file_name('metadata.json')), 'w') as f:
            json.dump({
                'n_pos_samples': n_pos_samples,
                'n_neg_samples': n_neg_samples,
            }, f)
        # with open(osp.join(kde_save_pth, 'label_id2name.json'), 'w') as f:
        #     json.dump({label_id: label_name for label_name, label_id in dep_parser_llama.labelmap.items()}, f)
        label_id2name = {label_id: label_name for label_name, label_id in labelmap.items()}
        with open(osp.join(kde_save_pth, 'label_names.txt'), 'w') as f:
            label_names = [label_id2name[label_id] for label_id in labelmap.values()]
            json.dump(label_names, f)
    return (pos_features, neg_features), height_features, label_features
