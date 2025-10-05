import torch.nn as nn
import os

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
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)   ## [bs, 35]，这里面padding为True其实会自动padding到这个batch中最长的这个text来算
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=bool) ## [bs, 35, 768]
        # output = output * mask.unsqueeze(-1)
        return output, mask
