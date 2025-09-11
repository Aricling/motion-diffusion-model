## 其实是finetune clip训练的启动代码,默认都是finetune CLIP所有的线性层，在这个库中会自动调用VAE
python -m train.train_finetune_clip \
    --save_dir save/0910_finetune_clip_using_VAE_all_layer_norm_try_0'eval_scale_problem' \
    --dataset humanml \
    --diffusion_steps 50 \
    --arch trans_dec \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 3 \
    --overwrite \
    --train_platform_type WandBPlatform \
    --pooling 1
    # --multiview

# eval fintuned CLIP启动代码，最后会报错，不过会运行成功的
python -m eval.eval_fted_clip \
    --model_path /home/mengqing/usr/motion-diffusion-model/save/0908_finetune_clip_all_layers_using_VAE_try1/model000250000.pt \
    --device 1


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