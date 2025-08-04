# 以微调的CLIP来训练MDM，启动代码
python -m train.train_mdm \
    --save_dir save/train_mdm_w_finetuned_clip_v0 \
    --dataset humanml \
    --diffusion_steps 50 \
    --arch trans_dec \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 3 \
    --overwrite \
    --train_platform_type WandBPlatform