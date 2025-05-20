batch_size=128
if [ $1 != "" ] ; then
    echo "new batch size: ""'"$1"'"
    batch_size=$1
fi
python explaination_baselines.py v_information \
    --train_data_dir ../pretrained-models/kde/open_llama_7b \
    --model_pth ../pretrained-models/kde/open_llama_7b_balanced/baselines/independent_mlp_h_2_4_leaky_relu_0.0_0.0_balanced/best_model.pt \
    --hidden_sizes "[2, 4]" --hidden_activation leaky_relu --batch_size ${batch_size}