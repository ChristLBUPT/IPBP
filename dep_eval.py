# -*- coding: utf-8 -*-
from typing import List, Mapping, Any, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import defaultdict
from io import StringIO


class Evaluator(object):

    def __init__(self, eps=1e-5):
        super(Evaluator, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        return f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        arc_mask = arc_preds.eq(arc_golds)[mask]
        rel_mask = rel_preds.eq(rel_golds)[mask] & arc_mask

        self.total += len(arc_mask)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_rels += rel_mask.sum().item()

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

def dict_sort(dict_to_sort: Mapping[Any, Any], **kwargs):
    return sorted(dict_to_sort.items(), **kwargs)

def dict_freq_sort(dict_to_sort: Mapping[Any, int]):
    return sorted(dict_to_sort.items(), key=lambda x: x[1], reverse=True)


class ResultProcessor:
    def __init__(self, result_file_name: str, dataset_file_name: str) -> None:
        self.result_file_name = result_file_name
        self.dataset_file_name = dataset_file_name
        self.results = self.process_file(self.result_file_name, self.process_result_line)
        self.golds = self.process_file(self.dataset_file_name, self.process_dataset_line)

    
    @staticmethod
    def process_result_line(line: str):
        return line.split('\t')

    @staticmethod
    def process_dataset_line(line: str):
        line_splt = line.split('\t')
        return [line_splt[1], line_splt[6], line_splt[7]]

    @staticmethod
    def process_file(result_file_name: str, process_method):
        results = []
        with open(result_file_name, 'r') as f:
            this_result = []
            for line in f.read().split('\n'):
                if line:
                    # tail, head, dep = process_method(line)
                    this_result.append(process_method(line))
                else:
                    # for i in range(len(this_result)):
                    #     head = int(this_result[i][1]) - 1
                    #     head = f"[{head}]{this_result[head][0]}" if head >= 0 else 'root'
                    #     this_result[i][1] = head
                    # for i in range(len(this_result)):
                    #     this_result[i][0] = f'[{i}]{this_result[i][0]}'
                    results.append(this_result)
                    this_result = []
        
        return results
    
    @staticmethod
    def visualize_dependency(sent: List[str], tail: int, head: int, rel: str):
        sent = ['root'] + sent
        sent_lens = [len(each) + 1 for each in sent]
        start_indices = [0] + np.cumsum(sent_lens).tolist()[:-1]
        l, r = (tail, head) if tail < head else (head, tail)
        arrow_start_pos = start_indices[l]
        arrow_end_pos = start_indices[r] + len(sent[r]) - 1
        arrow_length = arrow_end_pos - arrow_start_pos + 1
        up_line = ' ' * arrow_start_pos + '-' * arrow_length
        mid_line_1 = ' ' * arrow_start_pos + '|' + ' ' * (arrow_length - 2) + '|' + f' {rel}'
        mid_line_2 = ' ' * arrow_start_pos + ('|' if tail < head else '+') + ' ' * (arrow_length - 2) + ('+' if tail < head else '|')
        return '\n'.join([up_line, mid_line_1, mid_line_2, ' '.join(sent)])
    
    @staticmethod
    def visualize_dependency_mpl(sent: List[str], tail: int, head: int, rel: str, y: float = 1, skip_text: bool = False, estimated_fontsize: float = 0.04):
        # fig = plt.figure()
        starts = [1]

        for token in sent:
            if not skip_text:
                plt.text(starts[-1], y, token, )
            starts.append(starts[-1] + (len(token) + 2) * estimated_fontsize)
        
        plt.xlim(1, starts[-1])
        
        arrow_start = starts[tail] + estimated_fontsize * len(sent[tail]) / 2
        arrow_end = starts[head] + estimated_fontsize * len(sent[head]) / 2

        y_sig = y / abs(y)
        y += estimated_fontsize * 3 * y_sig
        split = 5
        first_split = arrow_start + (arrow_end - arrow_start) / split
        second_split = arrow_start + (split - 1) * (arrow_end - arrow_start) / split
        arc_deltay = abs(arrow_end - arrow_start) * 0.3 * y_sig
        plt.plot([arrow_start, first_split], [y, y + arc_deltay], color='black')
        plt.plot([first_split, second_split], [y + arc_deltay, y + arc_deltay], color='black')
        plt.text((first_split + second_split) / 2 - len(rel) * estimated_fontsize / 2, y + arc_deltay * 1.1, rel)
        plt.arrow(second_split, y + arc_deltay, (arrow_end - arrow_start) / split, -arc_deltay, color='red', head_width=0.005, head_length=0.01)
        # plt.show()
        # plt.savefig('test.png', dpi=400)
        # plt.close(fig)

    
    def get_wrong_case(self, silent: bool = False, **filter_kwargs):
        false_pred_labels2freq = defaultdict(int)
        false_gold_labels2freq = defaultdict(int)
        mistaken_pair2freq = defaultdict(int)
        
        total_labels2freq = defaultdict(int)

        allowed_result_deps = filter_kwargs.pop('result_deps', None)
        allowed_gold_deps = filter_kwargs.pop('gold_deps', None)
        allowed_dep_pairs = filter_kwargs.pop('dep_pairs', None) 
        # print(f'allowed_result_deps: {allowed_result_deps}, allowed_gold_deps: {allowed_gold_deps}, allowed_dep_pairs: {allowed_dep_pairs}')

        for result, gold in zip(self.results, self.golds):
            for gold_item in gold:
                total_labels2freq[gold_item[2]] += 1
            if result != gold:
                sentence = [each[0] for each in result]
                if not silent:
                    print('===============================================================')
                    print(f"sentence: {' '.join(sentence)}")

                for i, (result_item, gold_item) in enumerate(zip(result, gold), start=1):
                    if result_item != gold_item:
                        if not silent:
                            this_result_dep, this_gold_dep = result_item[2], gold_item[2]
                            if allowed_result_deps is not None and this_result_dep not in allowed_result_deps \
                                or allowed_gold_deps is not None and this_gold_dep not in allowed_gold_deps \
                                or allowed_dep_pairs is not None and (this_gold_dep, this_result_dep) not in allowed_dep_pairs:
                                continue
                            print('================')
                            print('result:')
                            print(self.visualize_dependency(sentence, i, int(result_item[1]), result_item[2]))
                            print('gold:')
                            print(self.visualize_dependency(sentence, i, int(gold_item[1]), gold_item[2]))
                        false_pred_labels2freq[result_item[2]] += 1
                        false_gold_labels2freq[gold_item[2]] += 1
                        mistaken_pair2freq[(gold_item[2], result_item[2])] += 1
        
        def add_frac_and_total(labels2freq_dict: Mapping[Union[str, Tuple[str, str]], int]):
            return {
                each: {
                    'frac': labels2freq_dict[each] / total_labels2freq[each if isinstance(each, str) else each[0]],
                    'cnt': labels2freq_dict[each],
                    'total': total_labels2freq[each if isinstance(each, str) else each[0]]
                } for each in labels2freq_dict
            }

        false_pred_labels2frac = add_frac_and_total(false_pred_labels2freq)
        false_gold_labels2frac = add_frac_and_total(false_gold_labels2freq)
        mistaken_pair2frac = add_frac_and_total(mistaken_pair2freq)
        
        if not any((allowed_dep_pairs, allowed_gold_deps, allowed_result_deps)):
            # print('false_pred_labels2freq'.center(90, '='))
            # print(*dict_freq_sort(false_pred_labels2freq), sep='\n')
            # print('false_gold_labels2freq'.center(90, '='))
            # print(*dict_freq_sort(false_gold_labels2freq), sep='\n')
            # print('mistaken_pair2freq'.center(90, '='), sep='\n')
            # print(*dict_freq_sort(mistaken_pair2freq))

            print('false_pred_labels2frac'.center(90, '='))
            print(*dict_sort(false_pred_labels2frac, key=lambda x: x[1]['cnt'], reverse=True), sep='\n')
            print('false_gold_labels2frac'.center(90, '='))
            print(*dict_sort(false_gold_labels2frac, key=lambda x: x[1]['cnt'], reverse=True), sep='\n')
            print('mistaken_pair2frac'.center(90, '='))
            print(*dict_sort(mistaken_pair2frac, key=lambda x: x[1]['cnt'], reverse=True), sep='\n')

    
    def get_wrong_case_mpl(self, savefig_dir: str = './visualization/'):
        os.system(f'rm {savefig_dir}/*')
        for idx, (result, gold) in enumerate(zip(tqdm(self.results), self.golds)):
            if result != gold:
                sentence = ['root'] + [each[0] for each in result]
                # image_fname = f"{idx:03d}_{' '.join(sentence)}.png"
                image_fname = f"{idx:03d}.png"

                text_placed = False
                sentence_length = sum([len(each) for each in sentence]) + 2 * len(sentence)
                fig = plt.figure(figsize=(sentence_length * 0.1, 4))
                for i, (result_item, gold_item) in enumerate(zip(result, gold), start=1):
                    if result_item != gold_item:
                        self.visualize_dependency_mpl(sentence, i, int(result_item[1]), result_item[2], 1, text_placed)
                        self.visualize_dependency_mpl(sentence, i, int(gold_item[1]), gold_item[2], -1, text_placed)
                        text_placed = True
                    
                plt.show()
                plt.savefig(os.path.join(savefig_dir, image_fname), dpi=400)
                plt.close(fig)
                

if __name__ == '__main__':
    rp = ResultProcessor('./models/bert-large-gcn-test_2024-03-22-22-17-44/dev_result.txt', './data/dev.conllu')
    # rp.get_wrong_case()
    # rp.visualize_dependency_mpl(['root', '123456789012345678901234567890', 'de', 'f'], 1, 3, 'test')
    # rp.get_wrong_case_mpl()
    # rp.get_wrong_case(silent=True)
    rp.get_wrong_case(dep_pairs=[('prep' ,'prep'), ])