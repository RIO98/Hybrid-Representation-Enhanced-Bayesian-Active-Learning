#!/bin/bash
simg_path="/win/your_path/your_singularity.simg"
project_dir="/win/your_path/here/Hybrid-Representation-Enhanced-Bayesian-Active-Learning"
exp_dir="./examples/us_thyroid_segmentation/preprocessed"
num_core=8
num_gpu=1
node="your_node"

seed=0
n_layers=4
n_classes=2
loss_func="focal"
method="simi_mi"
coef=0.5
exp_n="volume"
iteration=5000
valid_freq=500
increment=1
buffer_size=6
eval_file_name="bayesian_al_methods_vol.py"
data_root="${exp_dir}/us_thyroid"
_common_path="${exp_n}/seed_${seed}/${n_layers}layer/${method}"
txt_save_root="${exp_dir}/experiments/${_common_path}"
eval_save_root="${exp_dir}/eval/${_common_path}"

is=1
ie=5
flag=0
for ((i = is; i < ie; i++))
do
  _stage=${i}
  train_patients="${txt_save_root}/id-list_trial-${_stage}_training-0.txt"
  valid_patients="${txt_save_root}/id-list_trial-1_validation-0.txt"
  model_save_dir="${exp_dir}/logs/${_common_path}/stage${_stage}_iter${iteration}"
  pred_save_root="${exp_dir}/results/${_common_path}/stage${_stage}_iter${iteration}"
  slurm_out="${project_dir}/slurm_out"
  mkdir -p ${slurm_out}

  if [ ${flag} == 0 ]
  then
    jid_train=$(sbatch --parsable --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
    --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} \
    python train_bcnn_w_valid.py --data_root ${data_root} --train_patients ${train_patients} \
    --model_save_dir ${model_save_dir} --iteration ${iteration} --valid_freq ${valid_freq} --n_classes ${n_classes}\
    --loss_func ${loss_func} --valid_patients ${valid_patients} --n_layers ${n_layers} --seed ${seed}")
    echo "${jid_train}"
    flag=1
  else
    jid_train=$(sbatch --parsable --dependency=aftercorr:"${jid_select}" --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
    --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} \
    python train_bcnn_w_valid.py --data_root ${data_root} --train_patients ${train_patients} \
    --model_save_dir ${model_save_dir} --iteration ${iteration} --valid_freq ${valid_freq} --n_classes ${n_classes}\
    --loss_func ${loss_func} --valid_patients ${valid_patients} --n_layers ${n_layers} --seed ${seed}")
  fi


  testing_set="databank"
  slice_list="${txt_save_root}/id-list_trial-${_stage}_${testing_set}-0.txt"
  pred_save_dir="${pred_save_root}/${testing_set}"
  jid_databank=$(sbatch --parsable --dependency=aftercorr:"${jid_train}" --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
  --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} \
  python test.py --slice_list ${slice_list} --pred_save_dir ${pred_save_dir} --model_save_dir ${model_save_dir} \
  --data_root ${data_root} --n_layers ${n_layers} --n_classes ${n_classes}")

  jid_select=$(sbatch --parsable --dependency=aftercorr:"${jid_databank}" --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
  --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} python ${eval_file_name} \
  --stage ${_stage} --mode ${testing_set} --method ${method} --exp ${exp_n} --seed ${seed} --n_layers ${n_layers} \
  --coef ${coef} --iteration ${iteration} --increment ${increment} --buffer_size ${buffer_size} --n_classes ${n_classes} \
  --image_root ${data_root} --pred_save_root ${pred_save_root} --eval_save_root ${eval_save_root} --txt_save_root ${txt_save_root}")


  testing_set="testing"
  slice_list="${txt_save_root}/id-list_trial-1_${testing_set}-0.txt"
  pred_save_dir="${pred_save_root}/${testing_set}"
  jid_test=$(sbatch --parsable --dependency=aftercorr:"${jid_train}" --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
  --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} \
  python test.py --slice_list ${slice_list} --pred_save_dir ${pred_save_dir} --model_save_dir ${model_save_dir} \
  --data_root ${data_root} --n_layers ${n_layers} --n_classes ${n_classes}")

  sbatch --parsable --dependency=aftercorr:"${jid_test}" --no-requeue --gres=gpu:${num_gpu} -n ${num_core} -D ${project_dir} -w ${node} -o "${slurm_out}/%j_${node}.out" \
  --wrap="nvidia-smi && singularity exec --nv -B /win/salmon/user ${simg_path} python ${eval_file_name} \
  --stage ${_stage} --mode ${testing_set} --method ${method} --exp ${exp_n} --seed ${seed} --n_layers ${n_layers} \
  --iteration ${iteration} --n_classes ${n_classes} --image_root ${data_root} --pred_save_root ${pred_save_root} \
  --eval_save_root ${eval_save_root} --txt_save_root ${txt_save_root}"

  sleep 2s

done

