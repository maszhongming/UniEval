DATA_DIR=data/fact/qags_xsum.json

python predict_score.py \
    --task fact \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task fact \
    --dataset qags_xsum \

DATA_DIR=data/fact/qags_cnndm.json

python predict_score.py \
    --task fact \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task fact \
    --dataset qags_cnndm \

