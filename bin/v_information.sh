python explaination_baselines.py \
    independent_mlp \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --val_data_dir ../pretrained-models/kde/open_llama_7b_debug \
    --hidden_sizes "[2]" \
    --hidden_activation "leaky_relu" \
    --l1_lambda 0.0 \
    --l2_lambda 0.0 


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