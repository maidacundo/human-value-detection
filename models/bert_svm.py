import torch
import pytorch_lightning as pl
from transformers import AutoConfig, BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torchmetrics

class BertSVMmodel(pl.LightningModule):
    def __init__(self, model_name):
        #lr=2e-5, classifier_dropout=.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        
        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask
        )
        
        cls = outputs[0][:,0,:]
        pooler = outputs.pooler_output

        return cls, pooler
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cls, pooler = self(input_ids, attention_mask)

        return cls, pooler