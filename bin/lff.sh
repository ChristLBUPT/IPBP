python explaination_baselines.py \
    mlp \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
    --l1_lambda 0.0 \
    --l2_lambda 0.0 \
    --batch_size 4096 \
    --hidden_sizes "[512, 128]"

python explaination_baselines.py \
    mlp \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
    --hidden_sizes "[]" \
    --l1_lambda 0.01 \
    --l2_lambda 0.0 

python explaination_baselines.py \
    mlp \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
    --hidden_sizes "[]" \
    --l1_lambda 0.0 \
    --l2_lambda 0.01 

python explaination_baselines.py \
    mlp \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
    --hidden_sizes "[]" \
    --l1_lambda 0.01 \
    --l2_lambda 0.01 


# python explaination_baselines.py \
#     independent_mlp \
#     --train_data_dir ../pretrained-models/kde/open_llama_7b \
#     --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
#     --hidden_sizes "[2, 4, 8]" \
#     --hidden_activation "leaky_relu" \
#     --l1_lambda 0.0 \
#     --l2_lambda 0.0 


# python explaination_baselines.py \
#     independent_mlp \
#     --train_data_dir ../pretrained-models/kde/open_llama_7b \
#     --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
#     --hidden_sizes "[2, 4, 8, 16]" \
#     --hidden_activation "leaky_relu" \
#     --l1_lambda 0.0 \
#     --l2_lambda 0.0 


# python explaination_baselines.py \
#     independent_mlp \
#     --train_data_dir ../pretrained-models/kde/open_llama_7b \
#     --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
#     --hidden_sizes "[2]" \
#     --hidden_activation "leaky_relu" \
#     --l1_lambda 0.0 \
#     --l2_lambda 0.0 