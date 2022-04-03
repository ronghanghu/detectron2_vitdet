entitlement=gpu_pnb_fair
entitlement=fair_gpu_pnb
entitlement=default_pnb_gpu    #v100_32
#entitlement=gpu_fair
entitlement=default_prn_gpu  #a100


gpu=V100_32G
gpu=A100


prefix=""
n=8
configs=(
#configs/mae_vit/mask_rcnn_vit_b_w14_4res_50ep.py
configs/mae_vit/mask_rcnn_vit_l_w14_4res_50ep.py
)
resume=f325246042
for config in ${configs[@]}; do
  echo ${config}
for lr in 1e-4; do
# for lr in 0.0002 0.00032; do

 for ld in 0.7; do
   ../../dev/fb/launch.py ${config} ${prefix}${n}n_lr${lr}_ld${ld} --entitlement ${entitlement}  --manifold-bucket fair_logging --gpu-type ${gpu} --num-machines ${n} --target //vision/fair/detectron2/tools:lazyconfig_train_net --extra-flags "  optimizer.lr=${lr}" "  optimizer.params.lr_decay_rate=${ld}"

    # resume from
   # ../../dev/fb/launch.py ${config} ${prefix}${n}n_lr${lr}_ld${ld}_resume${resume} --resume-from ${resume}/model_0079999.pth --entitlement ${entitlement}  --manifold-bucket fair_logging --gpu-type ${gpu} --num-machines ${n} --target //vision/fair/detectron2/tools:lazyconfig_train_net --extra-flags "  optimizer.lr=${lr}" "  optimizer.params.lr_decay_rate=${ld}"

  # for bl in "single" "bottleneck2"; do # "conv3_1_g32" "convnext" "convnext" "conv3_1_g32" "single"
  #  ../../dev/fb/launch.py ${config} ${prefix}${n}n_bl${bl}_lr${lr}_ld${ld} --entitlement ${entitlement}  --manifold-bucket fair_logging --gpu-type ${gpu} --num-machines ${n} --target //vision/fair/detectron2/tools:lazyconfig_train_net --extra-flags "  optimizer.lr=${lr}" "  optimizer.params.lr_decay_rate=${ld}" "  model.backbone.bottom_up.net.residual_block=${bl} "
  #   ### resume from
  #   #../../dev/fb/launch.py ${config} ${prefix}${n}n_bl${bl}_lr${lr}_ld${ld}_resume${resume} --resume-from ${resume}/model_0079999.pth --entitlement ${entitlement}  --manifold-bucket fair_logging --gpu-type ${gpu} --num-machines ${n} --target //vision/fair/detectron2/tools:lazyconfig_train_net --extra-flags "  optimizer.lr=${lr}" "  optimizer.params.lr_decay_rate=${ld}" "  model.backbone.bottom_up.net.residual_block=${bl} "
  # done

 done


done
done
