import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torchmetrics

class BertClassifierTransferLearning(pl.LightningModule):
    def __init__(self, model_name, num_labels, classifier_dropout, optimizer, lr, n_training_steps=None, n_warmup_steps=None, num_layers_tl=3):
        #lr=2e-5, classifier_dropout=.1):
        super().__init__()

        self.optim = optimizer
        self.lr = lr
        self.classifier_dropout = classifier_dropout

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.classifier_dropout = classifier_dropout
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.transformer = AutoModel.from_pretrained(model_name)

        # freezing the first layers of the model
        if num_layers_tl != 0:
            for i in range(self.config.num_hidden_layers-num_layers_tl):
                for param in self.transformer.encoder.layer[i].parameters():
                    param.requires_grad = False
        elif num_layers_tl == 0:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.init_weights() # https://pytorch.org/docs/stable/nn.init.html
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        self.losses = []
        self.val_losses = []
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # verificare che cos'è il pooled output (in realtà conviene verificare che cos'è tutto l'output)
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        loss = 0
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), output, (hidden_states), (attentions)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels=labels)
        self.log("train_loss", outputs[0], prog_bar=True, logger=True)
        
        self.losses.append(outputs[0])
        return {"loss": outputs[0], "predictions": torch.sigmoid(outputs[1]), "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels=labels)
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.val_losses.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels=labels)
        loss = outputs[0]
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels=labels)
        return torch.sigmoid(outputs[1])

    def configure_optimizers(self):

        optimizer = self.optim([p for p in self.parameters() if p.requires_grad], lr=self.lr, weight_decay=1e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )