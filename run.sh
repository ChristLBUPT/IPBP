options=$(getopt -o "dp::ce:m:flu:" -l "debug,profile::,check,external:,multi:,freeze,lora,userargs:" -n run.sh -- $@)
if [[ "$?" -ne 0 ]] ; then
  exit 2
fi
eval set -- $options
num_epochs=8
# if ! [ -v external ] ; then
#   external=biaffine
# fi
external=biaffine
ds_config_file="configs/ds_config_stage2_offload.json"
extract_cmd_arg() {
  cmd_args_str=$1
  arg_name=$2
  echo ${cmd_args_str} | sed 's/--/\n--/g' | grep ${arg_name} | sed -e 's/--//g' -e "s/${arg_name}//g" -e 's/=//g'
}
while [[ -n $1 ]] ; do
  case $1 in 
    -d | --debug ) 
      if [[ $(hostname -I) =~ .*113.* ]] ; then
        debug="14514" 
      else
        debug="0.0.0.0:8417"
      launch_prog="python -m debugpy --listen ${debug} --wait-for-client"
      fi
      shift 
      ;;
    -p | --profile ) 
      digit_re="[0-9\.]"
      if [[ $2 =~ $digit_re ]] ; then
        profile=" --profile $2"
        shift 2
      else
        profile=" --profile 0.2"
        shift
      fi
      num_epochs=1
      ;;
    -c | --check )
      do_sanity_check=" --sanity_check"
      shift
      ;;
    -e | --external )
      if [[ -n $2 ]] ; then
        external=$2
        shift 2
      else
        echo "'-e' requires an argument"
        exit 2
      fi
      ;;
    -m | --multi ) 
      launch_cmd="accelerate launch --config_file configs/ds_multi.yaml dep_main_acc.py"
      if [[ $2 =~ [0-9][0-9]? ]] ; then
        file_content_replaced=$(cat configs/ds_multi.yaml | sed -e "s/machine_rank: [0-9]*/machine_rank: $2/g")
        echo "$file_content_replaced" > configs/ds_multi.yaml
        # for line in $(cat configs/ds_multi.yaml | sed -e "s/machine_rank: 0/machine_rank: $2/g") ; do
        #   echo "line '${line}'"
        # done
        # echo $config_buf > configs/ds_multi.yaml
      else
        echo "-m|--multi must be a number, not '$2'"
        exit 2
      fi
      shift 2
      ;;
    -f | --freeze )
      freeze=" --freeze_base_model --train_batch_size=32 --gas=8"
      ds_config_file="configs/ds_config_stage2.json"
      shift
      ;;
    -l | --lora )
      lora=" --lora_llm"
      ds_config_file="configs/ds_config_stage2.json"
      shift
      ;;
    -u | --userargs )
      userargs=$2
      shift 2
      ;;
    -- )
      shift
      ;;
    * )
      if [ -z "${config_set}" ] && [ -f "$1" ] ; then
        source $1
        config_set=1
        shift
      else
        echo "unrecognized positional argument '$1'"
        exit 2
        # userargs+=($1 )
      fi
      ;;
  esac
done

if [[ ${model_type} =~ bert.* ]] ; then
  [[ -n ${launch_prog} ]] && launch_cmd="${launch_prog} dep_main.py" || launch_cmd="python dep_main.py"
else
  if [[ -z $launch_cmd ]] ; then
    # launch_cmd="accelerate launch --config_file configs/ds_with_config.yaml dep_main_acc.py"
    # launch_cmd="deepspeed dep_main_ds.py --ds_config configs/ds_config_stage2_offload.json"
    if [[ -n ${launch_prog} ]] ; then
      torchrun_pth=$(which torchrun)
      prefix="${launch_prog} ${torchrun_pth}"
    else
      prefix="torchrun"
    fi
    launch_cmd="${prefix} --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 dep_main_ds.py --ds_config ${ds_config_file}"
  fi
fi

model_name="${model_type}-${external}"\
"$([[ -n ${freeze} ]] && echo "-freeze")"\
"$([[ -n ${lora} ]] && echo "-lora")-test"\
"$([ -v debug ] && echo "-debug")"\
"$([ -v profile ] && echo "-profile")"\
"$([ -v userargs ] && echo " ${userargs}" | tr -- '-= ' '_')"
http_proxy=http://172.18.214.116:6666 https_proxy=http://172.18.214.116:6666 NCCL_DEBUG=WARN eval ${launch_cmd} ${cmd_args} \
  --model_name="${model_name}" \
  ${profile}${do_sanity_check}${freeze}${lora} ${userargs} #--gcn_transpose_edge_scores