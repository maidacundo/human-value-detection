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

        # verificare che cos'è il pooled output (in realtà conviene verificare che cos'è tutto l'output)
        embeddings = outputs[0][:,0,:].numpy()

        return embeddings 
    
    def train_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids, attention_mask)

        return outputs