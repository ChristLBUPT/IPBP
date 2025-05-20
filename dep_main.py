from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

import accelerate
from modules.optimization import BertAdam
from modules.schedulers import LinearWarmUpScheduler
from torch.optim import AdamW

from tqdm import tqdm, trange
from dep_helper import get_word2id, get_vocab, get_label_list, get_pos2id
from dep_eval import Evaluator
from dep_model import DependencyParser
from line_profiler import LineProfiler
import datetime
import time
import pickle as pkl
import math


def human_readable_param_cnt(param_cnt: int):
    unit = ['', 'K', 'M', 'B']
    unit_idx = 0
    while param_cnt > 1024 and unit_idx < 3:
        param_cnt /= 1024
        unit_idx += 1
    
    return f"{param_cnt:.2f}{unit[unit_idx]}"

def train(args):

    # if not os.path.exists('./logs'):
    #     os.mkdir('./logs')

    # if args.model_name_prefix is None:
    #     raise Warning('model name is not specified, the model will NOT be saved!')
    now_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    logger = logging.getLogger(f"{__name__}_seed_{args.seed}")
    logger.setLevel(logging.INFO)

    # acc = accelerate.Accelerator()

    global output_model_dir

    if output_model_dir is None:

        output_model_dir = os.path.join('./models', now_time + '_' + args.model_name, f'seed_{args.seed}')

        # log_file_name = f"./logs/log-{args.model_name_prefix}-" + now_time

        # deleted_previous_logs = []
        # for each in os.listdir('./logs'):
        #     if each.startswith(f'log-{args.model_name_prefix}'):
        #         os.remove(f'./logs/{each}')
        #         deleted_previous_logs.append(each)
    else:
        output_model_base_dir, _ = os.path.split(output_model_dir)
        output_model_dir = os.path.join(output_model_base_dir, f'seed_{args.seed}')
            
    log_file_name = os.path.join(output_model_dir, 'run.log')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    file_handler = logging.FileHandler(log_file_name, 'w')
    file_handler.setFormatter(formatter)


    # has_console_handler = False
    # for each in logger.handlers:
    #     if isinstance(each, logging.StreamHandler):
    #         has_console_handler = True
    #         break

    # if not has_console_handler:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # logger.info(f'found prevous run logs: {deleted_previous_logs}, deleted them')
    # logger.info('arguments:\n  ' + '\n  '.join([f"{key}: {vars(args)[key]}" for key in vars(args)]))
    # logger.info('arguments:\n' + json.dumps(vars(args), indent=2))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    logger.info(f'setting seed to {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    vocab = get_vocab(args.train_data_path)
    word2id = get_word2id(args.train_data_path, do_lower_case=True, freq_threshold=2)
    pos2id = get_pos2id(args.train_data_path)
    logger.info('# of word in train: %d: ' % len(vocab))

    label_list = get_label_list(args.train_data_path, labels_to_add=['<bos>'] if args.use_gemma else ['[CLS]', '[SEP]'])
    logger.info('# of tag types in train: %d: ' % (len(label_list) - 3))
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    hpara = DependencyParser.init_hyper_parameters(args)
    dep_parser = DependencyParser(label_map, hpara, args.bert_model,
                                  word2id=word2id, pos2id=pos2id,
                                  from_pretrained=(not args.vanilla), logger=logger)

    train_examples = dep_parser.load_data(args.train_data_path)
    dev_examples = dep_parser.load_data(args.dev_data_path)

    if args.sanity_check:
        logger.info(f'running sanity check, only run on 5% of training data and 20% of validation data')
        train_examples = train_examples[: round(len(train_examples) * 0.05)]
        dev_examples = dev_examples[: round(len(dev_examples) * 0.2)]
    
    if args.profile:
        logger.info(
            'running profiling, only run on {profile_perc}% of training data and {profile_perc}% of validation data'.format(
                profile_perc=int(args.profile * 100)
            )
        )
        train_examples = train_examples[: round(len(train_examples) * args.profile)]
        dev_examples = dev_examples[: round(len(dev_examples) * args.profile)]

    convert_examples_to_features = dep_parser.convert_examples_to_features
    feature2input = dep_parser.feature2input
    get_loss = dep_parser.get_loss
    decode = dep_parser.decode
    save_model = dep_parser.save_model

    total_params = sum(p.numel() for p in dep_parser.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %s' % human_readable_param_cnt(total_params))

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        dep_parser.half()
    dep_parser.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        dep_parser = DDP(dep_parser)
    # elif n_gpu > 1:
    #     dep_parser = torch.nn.DataParallel(dep_parser)

    param_optimizer = list(dep_parser.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # if args.use_gcn or args.use_biaffine:
        # `use_gcn` has higher priority than `use_biaffine`
    if args.use_gcn:
        hi_lr_param_prefixes = ['gcn', 'edge_classifier', 'rel_classifier']
    elif args.use_biaffine:
        hi_lr_param_prefixes = ['mlp_arc', 'mlp_rel', 'arc_attn', 'rel_attn']
    else:
        hi_lr_param_prefixes = ['linear_arc', 'rel_classifier', 'bias']
    param_hi_lr = []
    param_others = []
    hi_lr_param_names = []
    for name, param in param_optimizer:
        if any([name.startswith(each) for each in hi_lr_param_prefixes]):
            param_hi_lr.append(param)
            hi_lr_param_names.append(name)
        else:
            param_others.append((name, param))
    
    param_optimizer = param_others


    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # if args.use_gcn or args.use_biaffine:
    optimizer_grouped_parameters.append({'params': param_hi_lr, 'weight_decay': 0.01, 'lr': args.random_param_lr})
    logger.info(f'random_parameters_percentage: {human_readable_param_cnt(sum([each.numel() for each in param_hi_lr]))}(random)/{human_readable_param_cnt(total_params)}(total)')
    logger.info(f'random parameter names: {", ".join(hi_lr_param_names)}')
    if args.fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(dep_parser, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, )
    best_epoch = -1
    best_dev_uas = -1
    best_dev_las = -1

    history = {'epoch': [], 'uas': [], 'las': []}

    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            dep_parser.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            if args.extract_intermediate_outputs:
                intermediate_outputs = []
                labels = []
                metadata = {"metabatch_sizes": []}
                meta_batch_idx = 0
                def dump_intermediate(intermediate, labels, meta_batch_idx):
                    with open(os.path.join(f'./pickles/intermediate_states/metabatch_{meta_batch_idx:03d}.pkl'), 'wb') as f:
                        pkl.dump((intermediate, labels), f)
            for step, start_index in enumerate(pbar := tqdm(range(0, len(train_examples), args.train_batch_size))):
                dep_parser.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue

                train_features = convert_examples_to_features(batch_examples)

                input_ids, input_mask, l_mask, eval_mask, arcs, rels, word_ids, pos_ids, ngram_ids, ngram_positions, \
                segment_ids, valid_ids = feature2input(device, train_features)

                if not args.extract_intermediate_outputs:
                    arc_scores, rel_scores = dep_parser(input_ids, segment_ids, input_mask, valid_ids, l_mask, word_ids,
                                                        pos_ids,
                                                        ngram_ids, ngram_positions, return_hidden_states_and_attentions=args.use_gcn)
                    l_mask[:, 0] = 0
                    loss = get_loss(arc_scores, rel_scores, arcs, rels, l_mask)

                    if np.isnan(loss.to('cpu').detach().numpy()):
                        raise ValueError('loss is nan!')
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                            scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                    
                    postfix_dict = {'Cur-loss': loss.item(), 'Avg-loss': tr_loss / (step + 1)}
                    # lrs = []
                    # for each in optimizer.param_groups:
                    #     group_step = optimizer.state[[*each['params']][0]]['step']
                    #     lrs.append(each['lr'] * each['schedule'].get_lr(group_step))
                    #     # lrs.append(each['lr'])
                        
                    # for idx, lr in enumerate(lrs):
                    #     postfix_dict[f'lr_{idx}'] = lr
                    pbar.set_postfix(postfix_dict)
                
                else:
                    intermediate_outputs.append(
                        dep_parser(
                            input_ids, segment_ids, input_mask, valid_ids, l_mask, 
                            word_ids, pos_ids, ngram_ids, ngram_positions, extract_hidden_states_and_attentions=True
                        )
                    )
                    labels.append((arcs.cpu(), rels.cpu(), l_mask.cpu()))
                    pbar.set_description(f'extracting intermediate outputs(metabatch {meta_batch_idx + 1}/{math.ceil(len(pbar) / args.intermediate_metabatch_size)})...')
                    if len(intermediate_outputs) == args.intermediate_metabatch_size:
                        dump_intermediate(intermediate_outputs, labels, meta_batch_idx)
                        meta_batch_idx += 1
                        intermediate_outputs = []
                        labels = []
                        metadata["metabatch_sizes"].append(args.intermediate_metabatch_size)

            if args.extract_intermediate_outputs:
                dump_intermediate(intermediate_outputs, labels, meta_batch_idx)

                metadata["metabatch_sizes"].append(len(intermediate_outputs))
                metadata["totals"] = sum(metadata["metabatch_sizes"])
                with open('./pickles/intermediate_states/metadata.json', 'w') as f:
                    # import json
                    json.dump(metadata, f)

                return

            dep_parser.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                prediction = {}
                logger.info('\n')
                evaluator = Evaluator()
                eval_examples = dev_examples
                dep_parser.eval()
                all_arcs, all_rels = [], []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                label_map[0] = '<unk>'
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]

                    eval_features = convert_examples_to_features(eval_batch_examples)

                    input_ids, input_mask, l_mask, eval_mask, arcs, rels, word_ids, pos_ids, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids = feature2input(device, eval_features)

                    with torch.no_grad():
                        arc_scores, rel_scores = dep_parser(
                            input_ids, segment_ids, input_mask, valid_ids, l_mask,
                            word_ids, pos_ids, ngram_ids, ngram_positions, return_hidden_states_and_attentions=args.use_gcn
                        )
                    l_mask[:, 0] = 0
                    arc_preds, rel_preds = decode(arc_scores, rel_scores, l_mask)
                    evaluator(arc_preds, rel_preds, arcs, rels, eval_mask)

                    lens = l_mask.sum(1).tolist()
                    all_arcs.extend(arc_preds[l_mask].split(lens))
                    all_rels.extend(rel_preds[l_mask].split(lens))

                all_arcs = [seq.tolist() for seq in all_arcs]
                all_rels = [[label_map[label_id] for label_id in seq.tolist()] for seq in all_rels]

                prediction['all_arcs'] = all_arcs
                prediction['all_rels'] = all_rels

                uas, las = evaluator.uas * 100, evaluator.las * 100
                report = '%s: Epoch: dev: UAS:%.2f, LAS:%.2f' % (epoch + 1, uas, las)
                logger.info(report)
                history['epoch'].append(epoch)
                history['uas'].append(uas)
                history['las'].append(las)

                if args.model_name is not None:

                    output_eval_file = os.path.join(output_model_dir, "dev_report.txt")
                    with open(output_eval_file, "a") as writer:
                        writer.write(report)
                        writer.write('\n')

                logger.info('\n')
                if history['las'][-1] > best_dev_las:
                    best_epoch = epoch + 1
                    best_dev_uas = history['uas'][-1]
                    best_dev_las = history['las'][-1]

                    num_of_no_improvement = 0

                    if args.model_name:
                        with open(os.path.join(output_model_dir, 'dev_result.txt'), "w") as writer:
                            all_arcs = prediction['all_arcs']
                            all_rels = prediction['all_rels']
                            for example, arcs, rels in zip(dev_examples, all_arcs, all_rels):
                                words = example.text_a.split(' ')
                                for word, arc, rel in zip(words, arcs, rels):
                                    line = '%s\t%s\t%s\n' % (word, arc, rel)
                                    writer.write(line)
                                writer.write('\n')

                        # model_to_save = dep_parser.module if hasattr(dep_parser, 'module') else dep_parser
                        best_eval_model_dir = os.path.join(output_model_dir, 'model')
                        if not os.path.exists(best_eval_model_dir):
                            os.makedirs(best_eval_model_dir)

                        save_model(best_eval_model_dir, args.bert_model)
                        arg_file = os.path.join(output_model_dir, 'args.json')
                        with open(arg_file, 'w', encoding='utf8') as f:
                            json.dump(vars(args), f, indent=2)
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

            if args.sanity_check:
                break

        best_report = "Epoch: %d, dev_UAS: %f, dev_LAS: %f" % (
            best_epoch, best_dev_uas, best_dev_las)

        logger.info("\n=======best las========")
        logger.info(best_report)
        logger.info("=======================")

        if args.model_name is not None:
            output_eval_file = os.path.join(output_model_dir, "final_report.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(str(args.model_name) + ' ')
                writer.write('total parameters: %d\n' % total_params)
                writer.write(best_report)
                writer.write('\n')

            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')
        
        return best_dev_uas, best_dev_las


def test(args):
    print(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    dep_parser = DependencyParser.load_model(args.eval_model, device)

    total_params = sum(p.numel() for p in dep_parser.parameters() if p.requires_grad)
    print('# of trainable parameters: %d' % total_params)

    eval_examples = dep_parser.load_data(args.test_data_path)

    convert_examples_to_features = dep_parser.convert_examples_to_features
    feature2input = dep_parser.feature2input
    decode = dep_parser.decode

    if args.fp16:
        dep_parser.half()
    dep_parser.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        dep_parser = DDP(dep_parser)
    elif n_gpu > 1:
        dep_parser = torch.nn.DataParallel(dep_parser)

    evaluator = Evaluator()
    dep_parser.eval()
    start_time = time.time()
    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]

        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, eval_mask, arcs, rels, word_ids, pos_ids, ngram_ids, ngram_positions, \
        segment_ids, valid_ids = feature2input(device, eval_features)

        with torch.no_grad():
            arc_scores, rel_scores = dep_parser(input_ids, segment_ids, input_mask, valid_ids, l_mask,
                                                word_ids, pos_ids,
                                                ngram_ids, ngram_positions)
        l_mask[:, 0] = 0
        arc_preds, rel_preds = decode(arc_scores, rel_scores, l_mask)
        evaluator(arc_preds, rel_preds, arcs, rels, eval_mask)

    total_time = time.time() - start_time
    sent_per_second = len(eval_examples) / total_time
    print('total time: %s seconds' % str(total_time))
    print('%s sentence per second' % str(sent_per_second))

    uas, las = evaluator.uas * 100, evaluator.las * 100
    report = '%s: UAS:%.2f, LAS:%.2f' % (args.test_data_path, uas, las)
    print(report)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default="./data/train.conllu",
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--dev_data_path",
                        default="./data/dev.conllu",
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--brown_data_path", default=None, type=str)
    parser.add_argument("--genia_data_path", default=None, type=str)
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_gemma",
                        action='store_true',
                        help="Whether to use Gemma.")
    parser.add_argument("--use_xlnet",
                        action='store_true',
                        help="Whether to use XLNet.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default="../pretrained-models/bert-large-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=300,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--word_pair_batch_size",
                        default=2048,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--random_param_lr", 
                        default=1e-3, 
                        type=float, 
                        help="The initial learning rate for random-initialized parameters " \
                        "(like bi-affine parameters and gcn parameters)")
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default=None)

    parser.add_argument('--patient', type=int, default=100, help="Patient for the early stop.")
    parser.add_argument('--model_name', type=str, default="default-model", help="directory used to store model checkpoints, eval results and hyperparameters")
    parser.add_argument('--mlp_dropout', type=float, default=0.33, help='')
    parser.add_argument('--n_mlp_arc', type=int, default=500, help='')
    parser.add_argument('--n_mlp_rel', type=int, default=100, help='')

    parser.add_argument("--use_biaffine", action='store_true')
    parser.add_argument("--dep_model", default=None, type=str,
                        help="")

    parser.add_argument("--vanilla", action='store_true')

    parser.add_argument("--use_pos", action='store_true')
    parser.add_argument("--use_encoder", action='store_true')
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--data_portion", type=int, default=None)
    parser.add_argument("--extract_intermediate_outputs", action='store_true')
    parser.add_argument("--intermediate_metabatch_size", type=int, default=32)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--profile", type=float, default=None)
    parser.add_argument("--use_gcn", action="store_true")
    parser.add_argument("--gcn_involve_node_emb", action="store_true")
    parser.add_argument("--gcn_transpose_edge_scores", action="store_true")
    parser.add_argument("--use_edge_gcn", action="store_true")

    args = parser.parse_args()
    print("non default arguments:")
    for action in parser._actions:
        if action.dest != 'help' and (arg_val := getattr(args, action.dest, None)) != action.default:
            print(f"  {action.dest}: {arg_val}")
    
    return args

def main(args):

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'RANK' in os.environ:
        assert args.local_rank == -1
        args.rank = int(os.environ['RANK'])
    elif 'SLURM_PROCID' in os.environ:
        assert args.local_rank == -1
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'LOCAL_RANK' in os.environ:
        assert args.local_rank == -1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        assert args.local_rank == -1
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    if args.init_method is None:
        if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
            args.init_method = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        else:
            args.init_method = "tcp://127.0.0.1:23456"

    args.local_rank = -1

    if args.do_train:
        return train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    args = get_args()

    global output_model_dir 
    output_model_dir = None
    if args.debug is not None:
        import debugpy
        print(f'debug mode, waiting for client at localhost:{args.debug}')
        debugpy.listen(args.debug)
        debugpy.wait_for_client()
    
    if args.profile:
        profiler = LineProfiler(train, DependencyParser.forward)
        profiler.run("main(args)")
        profiler.dump_stats("pickles/profiles/dep_main.py.lprof")
    else:
        # args.model_name_prefix = args.model_name
        uas_list, las_list = [], []
        for seed in [114, 514, 1919, 810, 1453]:
            args.seed = seed
            # args.model_name = f"{args.model_name_prefix}/seed-{seed}"
            run_best_uas, run_best_las = main(args)
            uas_list.append(run_best_uas)
            las_list.append(run_best_las)
        
        model_name_prefix, _ = os.path.split(os.path.abspath(output_model_dir))
        with open(os.path.join(model_name_prefix, 'stat_results.yaml'), 'w') as f:
            print(f'uas: {uas_list}', file=f)
            print(f'las: {las_list}', file=f)
            uas_arr = np.array(uas_list)
            las_arr = np.array(las_list)
            print(f'mean_uas: {uas_arr.mean().item():.4f}, std_uas: {uas_arr.std().item():.4f}')
            print(f'mean_las: {las_arr.mean().item():.4f}, std_las: {las_arr.std().item():.4f}')

            print(f'mean_uas: {uas_arr.mean().item():.4f}\nstd_uas: {uas_arr.std().item():.4f}', file=f)
            print(f'mean_las: {las_arr.mean().item():.4f}\nstd_las: {las_arr.std().item():.4f}', file=f)
            