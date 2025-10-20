import torch.nn as nn
import os
import z_config

def load_bert(model_path):
    bert = BERT(model_path)
    bert.eval()
    bert.text_model.training = False
    for p in bert.parameters():
        p.requires_grad = False
    return bert

class BERT(nn.Module):
    def __init__(self, modelpath: str):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)


    def forward(self, texts):
        if z_config.get_diy_config().model.MDM_model_file in ["mdm_29token_dim_test_projto32_rm_cls", "mdm_another_transformerEnclayer", "mdm_1_transEnc_same_cls", "mdm_29token_dim_test", "mdm_29token_dim_test_projto32"]:
            encoded_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",   # ✅ 固定长度 padding
                truncation=True,        # ✅ 超过自动截断
                max_length=50           # ✅ 固定长度 50
            )
        else:
            encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)   ## [bs, 35]，这里面padding为True其实会自动padding到这个batch中最长的这个text来算
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=bool) ## [bs, 35, 768]
        # output = output * mask.unsqueeze(-1)
        return output, mask
