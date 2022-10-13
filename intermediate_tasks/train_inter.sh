export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node 3 train_seq2seq.py \
    --model_name_or_path google/t5-v1_1-large \
    --do_train \
    --train_file data/intermediate_train.json \
    --text_column src \
    --summary_column tgt \
    --output_dir ./inter_model \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 16 \
    --save_strategy epoch \
    --num_train_epochs 10 \
    --ddp_find_unused_parameters False \
