import os
import json
from os.path import exists, join

def load_json(data_path):
    with open(data_path) as f:
        data = json.loads(f.read())
    return data

def write_predict(task, dataset, data, eval_scores):
    task_path = join('predict', task)
    if not exists(task_path):
        os.makedirs(task_path)
    write_path = join(task_path, '{}_result.json'.format(dataset))
    if exists(write_path):
        print("\nThe predicted scores are not saved because the result file already exists !!!")
    else:
        assert len(data) == len(eval_scores)
        for i in range(len(data)):
            data[i]['predict_scores'] = eval_scores[i]
        with open(write_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            print('\nPredicted scores are saved in {}'.format(write_path))
            
    
