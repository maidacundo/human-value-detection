import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torchmetrics

class BertBaselineClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, classifier_dropout, optimizer, lr, n_training_steps=None, n_warmup_steps=None, use_regularization=True):
        super().__init__()

        self.optim = optimizer
        self.lr = lr
        self.classifier_dropout = classifier_dropout

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Linear(768 * 3, num_labels)
        self.conclusion_attention = nn.Linear(768, 1)
        self.stance_attention = nn.Linear(768, 1)
        self.premise_attention = nn.Linear(768, 1)

        self.dropout = nn.Dropout(self.classifier_dropout)
        #self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.init_weights() # https://pytorch.org/docs/stable/nn.init.html
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        self.losses = []
        self.val_losses = []
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)

        self.use_regularization = use_regularization

    """
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
        reg_lambda=0.01
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
            # calculate L2 regularization term and add it to the loss
            if self.use_regularization:
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.parameters():
                    l2_reg += torch.linalg.vector_norm(param)
                loss += reg_lambda * l2_reg

            outputs = (loss,) + outputs

        return outputs  # (loss),  output, (hidden_states), (attentions)
    """
    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            reg_lambda=0.01):

        print(input_ids[:,0])
        print(attention_mask[:,0])
        print(token_type_ids[:,0])
        # Run the BERT model and get the hidden states
        _, conclusion_states = self.bert(input_ids[:, 0], attention_mask[:, 0], token_type_ids[:, 0])
        _, stance_states = self.bert(input_ids[:, 1], attention_mask[:, 1], token_type_ids[:, 1])
        _, premise_states = self.bert(input_ids[:, 2], attention_mask[:, 2], token_type_ids[:, 2])

        conclusion_attn = self.conclusion_attention(conclusion_states)
        stance_attn = self.stance_attention(stance_states)
        premise_attn = self.premise_attention(premise_states)

        # Compute softmax over the attention scores
        conclusion_attn = torch.softmax(conclusion_attn, dim=1)
        stance_attn = torch.softmax(stance_attn, dim=1)
        premise_attn = torch.softmax(premise_attn, dim=1)

        # Weight the hidden states by the attention scores
        conclusion_weighted_states = torch.bmm(conclusion_attn.unsqueeze(1), conclusion_states).squeeze(1)
        stance_weighted_states = torch.bmm(stance_attn.unsqueeze(1), stance_states).squeeze(1)
        premise_weighted_states = torch.bmm(premise_attn.unsqueeze(1), premise_states).squeeze(1)

        # Concatenate the weighted hidden states
        concat_states = torch.cat((conclusion_weighted_states, stance_weighted_states, premise_weighted_states), dim=1)
        concat_states = self.dropout(concat_states)
        logits = self.classifier(concat_states)

        # Calculate the loss and add L2 regularization
        outputs = (logits,) + outputs[2:]
        loss = 0
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fn(logits, labels)
            if self.use_regularization:
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.parameters():
                    l2_reg += torch.linalg.vector_norm(param)
                loss += reg_lambda * l2_reg
            outputs = (loss,) + outputs

        return outputs


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

        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=1e-5)

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