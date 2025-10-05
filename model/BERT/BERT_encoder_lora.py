import os
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def load_bert(model_path):
    bert = BERT(model_path)
    bert.eval()
    return bert

class BERT(nn.Module):
    def __init__(self, modelpath: str):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()

        # 避免 tokenizer 多线程 warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # 添加特殊 tokens
        special_tokens_dict = {
            "additional_special_tokens": ["<start_of_motion>", "<motion_token>", "<end_of_motion>"]
        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)

        # resize embedding 层，使得能处理新 token
        if num_added_toks > 0:
            self.text_model.resize_token_embeddings(len(self.tokenizer))

        # LoRA 配置
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],  # 子串匹配就行
        )
        self.text_model = get_peft_model(self.text_model, config)

    def forward(self, texts):
        # 编码输入
        encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 前向
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=torch.bool)
        return output, mask
