DATA_DIR=data/data2text/sfres.json

python predict_score.py \
    --task data2text \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task data2text \
    --dataset sfres \

DATA_DIR=data/data2text/sfhot.json

python predict_score.py \
    --task data2text \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task data2text \
    --dataset sfhot \

