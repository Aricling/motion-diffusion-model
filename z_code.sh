## 训练代码
# 以微调的CLIP来训练MDM，启动代码，并且加上了在训练的时候验证
python -m train.train_mdm \
    --save_dir save/0909_train_mdm_w_fted_clip_on_VAE_45w_try_0 \
    --dataset humanml \
    --diffusion_steps 50 \
    --arch trans_dec \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 2 \
    --overwrite \
    --train_platform_type WandBPlatform \
    --eval_during_training

先前的v含义：
v0: 通过cross attn来注入28维度的motion token内容
v1: 通过cross attn来注入1+28维度的text token和motion token内容
v2: text token来通过self attn来注入, motion token通过cross attn来注入?
True代表解决了clip的参数没有fix住的问题
# 继续训练代码，其中的权重文件代码里面会自动来找数字最大的
python -m train.train_mdm \
    --save_dir save/use_controlnet_to_inject_motion_tokens \
    --dataset humanml \
    --num_steps 600000 \
    --diffusion_steps 50 \
    --arch trans_enc \
    --text_encoder_type clip \
    --mask_frames \
    --use_ema \
    --device 2 \
    --overwrite \
    --eval_during_training \
    --train_platform_type WandBPlatform \
    --lr 5e-5

## eval代码
## 这个跑不了，后面返回会报错，gt指的是说明后面28个token都是使用的gt的
python -m eval.eval_humanml \
    --model_path /home/mengqing/usr/motion-diffusion-model/save/train_mdm_w_finetuned_clip_v1_True/model000200000.pt \
    --device 2

## 这个能跑,v1指的是以cross attn来进行信息注入的
python -m eval.eval_humanml \
    --model_path /home/mengqing/usr/motion-diffusion-model/save/train_mdm_w_finetuned_clip_v1_True/model000550000.pt \
    --device 2

## multiview的测试
python -m eval.eval_humanml \
    --model_path /home/mengqing/usr/motion-diffusion-model/save/train_mdm_w_multiview_tuned_CLIP/model000600000.pt \
    --device 4