## 其实是finetune clip训练的启动代码,在这里被改成了存储数据的方法
python -m train.train_mdm_w_motionbert \
    --save_dir save/test \
    --dataset humanml \
    --diffusion_steps 50 \
    --arch trans_dec \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 0 \
    --overwrite \
    # --train_platform_type WandBPlatform \
    --pooling 1 \
    --motionbert_config /home/mengqing/usr/motion-diffusion-model/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml \
    --evaluate_motionbert /home/mengqing/usr/motion-diffusion-model/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
    # --multiview

# eval fintuned CLIP启动代码
python -m eval.eval_humanml --model_path /home/mengqing/usr/motion-diffusion-model/save/finetune_clip_all__layers/lora000600000.pt


# 训来MDM+distilled Bert的代码
python -m train.train_mdm \
--save_dir save/my_humanml_trans_dec_bert_512 \
--dataset humanml \
--diffusion_steps 50 \
--arch trans_dec \
--text_encoder_type bert \
--mask_frames \
--use_ema \
--device 0