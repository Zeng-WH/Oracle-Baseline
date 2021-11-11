from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from rouge import Rouge
from utils import _get_word_ngrams
import re
from utils import cal_rouge

def lead_selection(doc_sent_list, lead_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    select_sents = []
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    if len(sents) < lead_size:
        select_sents = sents
    else:
        for i in range(lead_size):
            select_sents.append(sents[i])
    return select_sents

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            sort_sum = []
            for item in selected:
                sort_sum.append(doc_sent_list[item])
            return sort_sum
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    sort_selected = sorted(selected)
    sort_sum = []
    for item in sort_selected:
        sort_sum.append(doc_sent_list[item])

    return sort_sum


def main():
    with open('/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/test.source', 'r') as r:
        source_text = r.readlines()

    with open('/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/test.target', 'r') as r:
        target_text = r.readlines()
    tgt_list = []
    oracle_list = []
    for s, t in zip(source_text, target_text):
        s_sents = sent_tokenize(s)
        t_sents = sent_tokenize(t)
        s_tokens = []

        t_tokens = []
        for s_sent in s_sents:
            s_tokens.append(word_tokenize(s_sent))
        for t_sent in t_sents:
            t_tokens.append(word_tokenize(t_sent))
        sent_text = lead_selection(s_tokens, 3)
        oracle_txt = '<q>'.join([' '.join(tt) for tt in sent_text])
        tgt_txt = '<q>'.join([' '.join(tt) for tt in t_tokens])
        tgt_list.append(tgt_txt)
        oracle_list.append(oracle_txt)
    with open('/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/lead/train.tgt', 'w') as w:
        for item in tgt_list:
            w.write(item)
            w.write('\n')

    with open('/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/lead/train.oracle', 'w') as w:
        for item in oracle_list:
            w.write(item)
            w.write('\n')



if __name__ == '__main__':
    main()