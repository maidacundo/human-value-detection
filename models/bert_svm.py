import pytorch_lightning as pl
from transformers import BertModel
import torch.nn as nn

class BertSVMmodel(pl.LightningModule):
    def __init__(self, model_name):
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

        return (cls, pooler)
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        predictions = self(input_ids, attention_mask)

        return predictions[0], predictions[1]