export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 train_seq2seq.py \
    --model_name_or_path MingZhong/unieval-intermediate \
    --do_train \
    --train_file data/summarization/coherence_3w.json \
    --text_column src \
    --summary_column tgt \
    --output_dir ./continual_summ_coherence \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 6 \
    --max_source_length 1024 \
    --max_target_length 16 \
    --save_strategy steps \
    --save_steps 500 \
    --num_train_epochs 3 \
    --ddp_find_unused_parameters False \
