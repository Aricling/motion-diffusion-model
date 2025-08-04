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