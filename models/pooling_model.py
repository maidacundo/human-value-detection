import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

class TransformerClassifierPooling(pl.LightningModule):
    def __init__(self,model_name, num_labels, classifier_dropout, optimizer, lr, num_lstm_layers=1, n_training_steps=None, n_warmup_steps=None):
        super().__init__()

        self.optim = optimizer
        self.lr = lr
        self.classifier_dropout = classifier_dropout
        self.num_lstm_layers = num_lstm_layers

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.classifier_dropout = classifier_dropout
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.num_labels, batch_first=True, bidirectional=False, dropout=classifier_dropout, num_layers=self.num_lstm_layers)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # self.pooling_dense = nn.Linear(self.config.num_labels, self.config.num_labels)
        # self.pooling_activation = nn.Tanh()
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.init_weights() # https://pytorch.org/docs/stable/nn.init.html
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        for param in self.bert.parameters():
            param.requires_grad = False

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

        last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        last_hidden_state = self.dropout(last_hidden_state)
        lstm_output, _ = self.lstm(last_hidden_state)
        lstm_output = lstm_output.transpose(1, 2) # Convert from [batch_size, seq_len, hidden_size] to [batch_size, hidden_size, seq_len]

        avg_pooling = self.avg_pooling(lstm_output)
        avg_pooling = avg_pooling.view(avg_pooling.size(0), -1) # Flatten the tensor to [batch_size, hidden_size]

        max_pooling = self.max_pooling(lstm_output)
        max_pooling = max_pooling.view(max_pooling.size(0), -1) # Flatten the tensor to [batch_size, hidden_size]
        
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.classifier(pooled_output)

        # avg_pooling = self.pooling_dense(avg_pooling)
        # avg_pooling = self.pooling_activation(avg_pooling)
        avg_pooling = self.dropout(avg_pooling)

        # max_pooling = self.pooling_dense(max_pooling)
        # max_pooling = self.pooling_activation(max_pooling)
        max_pooling = self.dropout(max_pooling)

        logits = pooled_output + max_pooling + avg_pooling

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

        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=0.01)

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