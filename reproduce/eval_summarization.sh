DATA_DIR=data/summarization/summeval.json

python predict_score.py \
    --task summarization \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task summarization \
    --dataset summeval \
