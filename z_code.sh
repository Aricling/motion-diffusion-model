# 启动训练的代码，架构需要自己调整
python -m train.train_mdm --save_dir save/humanml_trans_dec_512_50steps_new \
    --dataset humanml \
    --diffusion_steps 50 \
    --mask_frames \
    --use_ema \
    --device 6 \
    --overwrite \
    --arch trans_dec \
    --train_platform_type WandBPlatform