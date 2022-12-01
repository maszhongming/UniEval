import sys
import json
import argparse
sys.path.append("..")
from utils import convert_to_json
from metric.evaluator import get_evaluator

def load_src(src_path):
    src_list = []
    with open(src_path) as f:
        for line in f:
            data = json.loads(line)
            src_list.append(data['src'])
    return src_list

def load_ref(ref_path):
    ref_list = []
    with open("reference-file.jsonl")as f:
        for line in f:
            data = json.loads(line)
            ref_list.append(data['ref'][0])
    return ref_list

def load_output(output_path):
    output_list = []
    with open("generator-output.jsonl") as f:
        for line in f:
            data = json.loads(line)
            output_list.append(data['hyp'])
    return output_list

def evaluate(args):
    # load data
    src_list = load_src(args.src_path)
    ref_list = load_ref(args.ref_path)
    output_list = load_output(args.hyp_path)

    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, 
                           src_list=src_list, ref_list=ref_list)

    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task=args.task, 
                              max_length=args.max_source_length,
                              device=args.device,
                              cache_dir=args.cache_dir)

    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, print_result=False)

    # Write predicted scores for all dimensions
    dims = ['fluency', 'coherence', 'consistency', 'relevance', 'overall']
    for dim in dims:
        with open('output_scores_{}.txt'.format(dim), 'w') as f:
            for i in range(len(eval_scores)):
                print(eval_scores[i][dim], file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get evaluation scores from UniEval from different NLG tasks'
    )

    parser.add_argument('--src_path', required=True,
        help='Path to the source files', type=str)
    parser.add_argument('--ref_path', required=True,
        help='Path to the reference files', type=str)
    parser.add_argument('--hyp_path', required=True,
        help='Path to the generated files', type=str)
    parser.add_argument('--task', default='summarization',
        help='Specific NLG task to be evaluated', type=str)
    parser.add_argument('--cache_dir', default=None,
        help='Where to store the pretrained models downloaded from huggingface.co', type=str)
    parser.add_argument('--device', default='cuda:0',
        help='Available device for the calculations', type=str)
    parser.add_argument('--max_source_length', default=1024,
        help='The maximum total input sequence length after tokenization', type=int)

    args = parser.parse_args()

    evaluate(args)