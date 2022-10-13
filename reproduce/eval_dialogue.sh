DATA_DIR=data/dialogue/topical_chat.json

python predict_score.py \
    --task dialogue \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task dialogue \
    --dataset topical_chat \
