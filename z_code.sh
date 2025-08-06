# 以微调的CLIP来训练MDM，启动代码
python -m train.train_mdm \
    --save_dir save/train_mdm_w_finetuned_clip_v1_True \
    --dataset humanml \
    --diffusion_steps 50 \
    --arch trans_dec \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 2 \
    --overwrite \
    --train_platform_type WandBPlatform

v0: 通过cross attn来注入28维度的motion token内容
v1: 通过cross attn来注入1+28维度的text token和motion token内容
v2: text token来通过self attn来注入, motion token通过cross attn来注入?
True代表解决了clip的参数没有fix住的问题

## eval代码
python -m eval.eval_humanml \
    --model_path /home/mengqing/usr/motion-diffusion-model/save/train_mdm_w_finetuned_clip_v1_True/model000450000.pt \
    --device 7