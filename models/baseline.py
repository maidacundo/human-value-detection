import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torchmetrics

class TransformerClassifier(pl.LightningModule):
    def __init__(
            self, 
            model_name, 
            num_labels, 
            classifier_dropout, 
            optimizer, 
            transfer_learning=False,
            lr_transformer=2e-5, 
            lr_classifier=1e-3, 
            weight_decay=1e-5, 
            n_training_steps=None, 
            n_warmup_steps=None,
        ):
        super().__init__()

        self.optim = optimizer
        self.lr_transformer = lr_transformer
        self.lr_classifier = lr_classifier
        self.weight_decay = weight_decay
        
        self.classifier_dropout = classifier_dropout

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.bert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.init_weights() # https://pytorch.org/docs/stable/nn.init.html
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        if transfer_learning:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.losses = []
        self.val_losses = []
        self.metrics = nn.ModuleDict({
            'accuracy': torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=.5),
            'f1': torchmetrics.F1Score(task="multilabel", num_labels=num_labels, average='macro', threshold=.5)
        })

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
        output_hidden_states=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
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

        return outputs  # (loss),  output, (hidden_states), (attentions)
    
    def compute_metrics(self, out, batch):
        labels = batch["labels"]
        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric(out, labels)
        return metrics

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
        preds = torch.sigmoid(outputs[1])
        metrics = self.compute_metrics(preds, batch)
        self.log_dict({f'val_loss': loss, **
                       {f'val_{k}': v for k, v in metrics.items()}})
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

        optimizer = self.optim([
                                    {"params": self.bert.parameters(), "lr": self.lr_transformer},
                                    {"params": self.classifier.parameters(), "lr": self.lr_classifier},
                                ],
                                lr=self.lr_transformer, 
                                weight_decay=self.weight_decay,
                                )

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