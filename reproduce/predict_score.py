import os
import sys
import argparse
from data_utils import load_json, write_predict
sys.path.append("..")
from metric.evaluator import get_evaluator

def predict(args, save_result=True):
    # load standard meta-evaluation benchmark
    data = load_json(args.data_path)

    # Initialize the evaluator for a specific task
    evaluator = get_evaluator(task=args.task, 
                              max_length=args.max_source_length,
                              device=args.device,
                              cache_dir=args.cache_dir)

    # get the evaluation scores for all the dimensions
    eval_scores = evaluator.evaluate(data)

    # save results with predicted scores
    if save_result == True:
        dataset = os.path.basename(args.data_path[:-5]) # get the name of dataset (w/o '.json')
        write_predict(args.task, dataset, data, eval_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get evaluation scores from UniEval from different NLG tasks'
    )

    parser.add_argument('--data_path', required=True,
        help='Path to the meta-evaluation benchmark', type=str)
    parser.add_argument('--task', required=True,
        help='Specific NLG task to be evaluated', type=str)
    parser.add_argument('--cache_dir', default=None,
        help='Where to store the pretrained models downloaded from huggingface.co', type=str)
    parser.add_argument('--device', default='cuda:0',
        help='Available device for the calculations', type=str)
    parser.add_argument('--max_source_length', default=1024,
        help='The maximum total input sequence length after tokenization', type=int)

    args = parser.parse_args()

    predict(args)
