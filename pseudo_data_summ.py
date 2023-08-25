import json
import copy
from tqdm import tqdm
import random
import numpy as np
from rank_bm25 import BM25Okapi
from nltk import sent_tokenize
from utils import fast_rouge, get_dec_and_ref

data_path = '/path/to/cnndm_train.jsonl'

def load_data(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Generate disfluent data. 1 positive sample corresponds to n_neg negative samples.
# Each negative sample contains n_noise disfluent noises
def disfluency_transformation(data, n_neg=3, n_noise=1):
    new_data = []
    for i in tqdm(range(len(data))):
        cur_sample = {}
        ### reference summary as groundtruth
        # cur_sample['src'] = data[i]['src']
        # cur_sample['tgt'] = ' '.join(data[i]['tgt'])
        ### lead 3 sentences as groundtruth
        cur_src = sent_tokenize(data[i]['src'])
        cur_sample['src'] = ' '.join(cur_src[3:])
        cur_sample['tgt'] = ' '.join(cur_src[:3])
        cur_sample['disfluent_tgt'] = []
        # j-th negative sample for i-th data
        for j in range(n_neg):
            ### reference summary as groundtruth
            # cur_tgt = (' '.join(data[i]['tgt'])).split()
            cur_tgt = (' '.join(cur_src[:3])).split()
            # add k noises
            for k in range(n_noise):
                tgt_len = len(cur_tgt)
                # length of span for transformation. Sampled from poisson distribution.
                span_len = min(tgt_len, np.random.poisson(5, 1)[0])
                # 1: insert, 2: delete, 3: shuffle
                transform_type = random.randint(1, 3)
                start_idx = random.randint(0, tgt_len - span_len)
                if transform_type == 1:
                    copy_idx = random.randint(0, tgt_len - span_len)
                    cur_tgt = cur_tgt[:start_idx] + cur_tgt[copy_idx:copy_idx+span_len] + cur_tgt[start_idx:]
                elif transform_type == 2:
                    cur_tgt = cur_tgt[:start_idx] + cur_tgt[start_idx+span_len:]
                elif transform_type == 3:
                    shuffled_span = cur_tgt[start_idx:start_idx+span_len]
                    random.shuffle(shuffled_span)
                    cur_tgt = cur_tgt[:start_idx] + shuffled_span + cur_tgt[start_idx+span_len:]
            cur_tgt = ' '.join(cur_tgt)
            cur_sample['disfluent_tgt'].append(cur_tgt)
        new_data.append(cur_sample)
    return new_data
            
# Generate incoherent data. 1 positive sample corresponds to n_neg negative samples.
# Each negative sample contains n_noise incoherent sentences
# retrieved path: processed data containing bm25_rankning
def incoherence_transformation(data, n_neg=3, n_noise=1, retrieved_path=None):
    if retrieved_path == None:
        corpus = []
        for i in range(len(data)):
            corpus.append(data[i]['src'].split())
        bm25 = BM25Okapi(corpus)
        for i in tqdm(range(len(data))):
            query = corpus[i]
            scores = bm25.get_scores(query)
            retrieved_index = np.flip(np.argsort(scores)).tolist()
            cur = {}
            cur['src'] = data[i]['src']
            cur['tgt'] = data[i]['tgt']
            cur['bm25_ranking'] = retrieved_index[:100]
            ### write data
            # with open('/path/to/cnndm/train_with_bm25.jsonl', 'a') as f:
            #     print(json.dumps(cur), file=f)
    else:
        data_with_bm25 = load_data(retrieved_path)
        new_data = []
        for i in tqdm(range(len(data))):
            cnt = 0
            # irrelevant_tgt = []
            incoherent_tgt = []
            cur_src = sent_tokenize(data[i]['src'])
            for idx in data_with_bm25[i]['bm25_ranking']:
                if idx == i or data[idx]['src'] == data[i]['src']:
                    continue
                '''
                # for reference summary
                cur_n = min(n_noise, len(data[i]['tgt']))
                cur_n = min(cur_n, len(data[idx]['tgt']))
                old_idx = random.sample(range(0, len(data[i]['tgt'])), cur_n)
                new_idx = random.sample(range(0, len(data[idx]['tgt'])), cur_n)
                cur_tgt = copy.deepcopy(data[i]['tgt'])
                for j in range(cur_n):
                    cur_tgt[old_idx[j]] = data[idx]['tgt'][new_idx[j]]
                '''
                # for lead 3
                cur_n = min(n_noise, 3)
                cur_tgt = copy.deepcopy(cur_src[:3])
                retrieved_tgt = sent_tokenize(data[idx]['src'])[:3]
                old_idx = random.sample(range(0, len(cur_tgt)), cur_n)
                new_idx = random.sample(range(0, len(retrieved_tgt)), cur_n)
                for j in range(cur_n):
                    cur_tgt[old_idx[j]] = retrieved_tgt[new_idx[j]]
                # irrelevant_tgt.append(' '.join(cur_tgt))
                incoherent_tgt.append(' '.join(cur_tgt))
                cnt += 1
                if cnt == n_neg:
                    break
            cur = {}
            cur['src'] = ' '.join(cur_src)
            cur['tgt'] = ' '.join(cur_src[:3])
            cur['gold_summary'] = data[i]['tgt']
            cur['incoherent_tgt'] = incoherent_tgt
            new_data.append(cur)
        return new_data

# Generate irrelevant data. 1 positive sample corresponds to n_neg negative samples.
# retrieved path: processed data containing bm25_rankning
def irrelevance_transformation(data, n_neg=3, retrieved_path=None):
    data_with_bm25 = load_data(retrieved_path)
    new_data = []
    for i in tqdm(range(len(data))):
        cnt = 0
        irrelevant_tgt = []
        cur_src = sent_tokenize(data[i]['src'])
        for idx in data_with_bm25[i]['bm25_ranking']:
            if idx == i or data[idx]['tgt'] == data[i]['tgt']:
                continue
    
            retrieved_tgt = sent_tokenize(data[idx]['src'])[:3] # negative samples
            irrelevant_tgt.append(' '.join(retrieved_tgt))
            cnt += 1
            if cnt == n_neg:
                break
        cur = {}
        cur['src'] = data[i]['src']
        cur['tgt'] = ' '.join(cur_src[:3]) # positive samples
        cur['gold_summary'] = data[i]['tgt'] # gold summary
        cur['irrelevant_tgt'] = irrelevant_tgt
        new_data.append(cur)
    return new_data

def main():
    # load data
    data = load_data(data_path)
    # process data for relevance dimension
    new_data = irrelevance_transformation(data, retrieved_path='/path/to/cnndm/train_with_bm25.jsonl')
    # write new data
    with open('/path/to/new_data.jsonl', 'w') as f:
        for i in range(len(new_data)):
            print(json.dumps(new_data[i]), file=f)

if __name__ == "__main__":
    main()
