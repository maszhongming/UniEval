import argparse
from os.path import join
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
from data_utils import load_json

def calculate_correlation(pred_score, human_score, dim, result):
    assert len(pred_score) == len(human_score)
    if dim not in result:
        result[dim] = [0] * 3
    result[dim][0] += pearsonr(pred_score, human_score)[0]
    result[dim][1] += spearmanr(pred_score, human_score)[0]
    result[dim][2] += kendalltau(pred_score, human_score)[0]
    return result

def print_correlations(result):
    table = PrettyTable(['Dimensions','Pearson', 'Spearman', 'Kendall'])
    for dim in result:
        table.add_row([dim, round(result[dim][0], 6), round(result[dim][1], 6), 
                            round(result[dim][2], 6)])
    print(table)

def get_unique_value(data, key):
    """
        Get a list of unique values for a specific key in the data.
    """
    value = set()
    for i in range(len(data)):
        if data[i][key] not in value:
            value.add(data[i][key])
    return list(value)

def correlation_for_summ(data, overall=True):
    """
        Provides calculation results of correlation at sample level, summary level and system level.
        For the specific definitions, please refer to the paper: https://arxiv.org/abs/2010.07100
    """
    dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
    if overall == True:
        dimensions.append('overall')

    # sample level correlation
    print('\n ********** Sample Level Correlations *********')
    result = {}
    for dim in dimensions:
        pred_score, human_score = [], []
        for i in range(len(data)):
            pred_score.append(data[i]['predict_scores'][dim])
            human_score.append(data[i]['scores'][dim])
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)
    
    # summary level correlation
    print('\n ********* Summary Level Correlations *********')
    result = {}
    docs = get_unique_value(data, 'doc_id')
    for dim in dimensions:
        valid_cnt = 0
        for doc_idx in docs:
            pred_score, human_score = [], []
            for i in range(len(data)):
                if data[i]['doc_id'] == doc_idx:
                    pred_score.append(data[i]['predict_scores'][dim])
                    human_score.append(data[i]['scores'][dim])
            if len(set(pred_score)) == 1 or len(set(human_score)) == 1:
                continue
            result = calculate_correlation(pred_score, human_score, dim, result)
            valid_cnt += 1
        for j in range(3):
            result[dim][j] /= valid_cnt
    print_correlations(result)
                
    # system level correlations
    print('\n ********** System Level Correlations *********')
    result = {}
    systems = get_unique_value(data, 'system_id')
    for dim in dimensions:
        pred_score, human_score = [], []
        for system_idx in systems:
            doc_cnt = 0
            cur_pred, cur_human = 0, 0
            for i in range(len(data)):
                if data[i]['system_id'] == system_idx:
                    cur_pred += data[i]['predict_scores'][dim]
                    cur_human += data[i]['scores'][dim]
                    doc_cnt += 1
            pred_score.append(cur_pred / doc_cnt)
            human_score.append(cur_human / doc_cnt)
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)
    

def correlation_for_dialog(data, overall=True):
    """
        Calculate turn-level correlation for dialogue response generation.
    """
    dimensions = ['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']
    if overall == True:
        dimensions.append('overall')

    # turn level correlation
    print('\n ************** Turn Level Correlations *************')
    result = {}
    for dim in dimensions:
        pred_score, human_score = [], []
        for i in range(len(data)):
            pred_score.append(data[i]['predict_scores'][dim])
            human_score.append(data[i]['scores'][dim])
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)
    

def correlation_for_d2t(data, overall=True):
    """
        Calculate sample-level correlation for data-to-text.
    """
    dimensions = ['naturalness', 'informativeness']
    if overall == True:
        dimensions.append('overall')

    # sample level correlation
    print('\n ************ Sample Level Correlations ***********')
    result = {}
    for dim in dimensions:
        pred_score, human_score = [], []
        for i in range(len(data)):
            pred_score.append(data[i]['predict_scores'][dim])
            human_score.append(data[i]['scores'][dim])
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)

def correlation_for_fact(data):
    """
        Calculate sample-level factual consistency score.
    """
    dim = 'consistency'

    # sample level correlation
    print('\n ********** Sample Level Correlations *********')
    result = {}
    pred_score, human_score = [], []
    for i in range(len(data)):
        pred_score.append(data[i]['predict_scores'][dim])
        human_score.append(data[i]['scores'][dim])
    result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)

def main(args):
    data_path = join(join('predict', args.task), '{}_result.json'.format(args.dataset))
    print('\nCorrelations for \'{}\' are shown below:'.format(data_path))
    data = load_json(data_path)
    if args.task == 'summarization':
        correlation_for_summ(data)
    elif args.task == 'dialogue':
        correlation_for_dialog(data)
    elif args.task == 'data2text':
        correlation_for_d2t(data)
    else:
        correlation_for_fact(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate the correlations between predicted scores and human scores'
    )

    parser.add_argument('--task', required=True,
        help='Specific NLG task to be evaluated', type=str)
    parser.add_argument('--dataset', required=True,
        help='The name of the meta-evaluation benchmark', type=str)

    args = parser.parse_args()
    assert args.task in ['summarization', 'dialogue', 'data2text', 'fact']

    main(args)
