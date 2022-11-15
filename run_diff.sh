#CUDA_VISIBLE_DEVICES=0 python main.py   --model klue/bert-base --generator_name klue/bert-base  --multi_gpu True   --test False   --max_len 64   --batch_size 128   --epochs 1   --eval_steps 250   --lr 0.00005   --masking_ratio 0.15   --lambda_weight 0.005   --warmup_ratio 0.05   --temperature 0.05   --path_to_data Dataset/   --train_data wiki_corpus.txt  --valid_data valid_sts.tsv --ckpt mask_15_lr_00005_lw_005_max64_bch128_bert_250.pt

CUDA_VISIBLE_DEVICES=0 python main.py   --model klue/roberta-base --generator_name klue/roberta-small  --multi_gpu True   --test False   --max_len 64   --batch_size 128   --epochs 1   --eval_steps 125   --lr 0.00005   --masking_ratio 0.15   --lambda_weight 0.005   --warmup_ratio 0.05   --temperature 0.05   --path_to_data Dataset/   --train_data wiki_corpus.txt  --valid_data valid_sts.tsv --ckpt best_checkpoint.pt