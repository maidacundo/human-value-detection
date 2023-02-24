import optuna
import torch
import pytorch_lightning as pl
from utils.data import HumanValuesDataModule
from models.baseline import BertBaselineClassifier

class HyperparameterTuner:
    def __init__(self, train_df, val_df, test_df, tokenizer, model_name, num_labels,
                 total_training_steps, warmup_steps, logger, early_stopping_callback):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_labels = num_labels
        self.total_training_steps = total_training_steps
        self.warmup_steps = warmup_steps
        self.logger = logger
        self.early_stopping_callback = early_stopping_callback

    def objective(self, trial):
        lr = trial.suggest_uniform("lr", 1e-6, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [8])
        optimizer = trial.suggest_categorical("optim", [torch.optim.Adam, torch.optim.AdamW])
        classifier_dropout = trial.suggest_categorical("optimizer_dropout", [.1, .2, .3])
        max_tok_len = trial.suggest_categorical("max_tok_len", [128, 256, 300, 368])

        config = {
            "lr": lr,
            "optim": optimizer,
            "classifier_dropout": classifier_dropout
        }

        data_module = HumanValuesDataModule(self.train_df, self.val_df, self.test_df, self.tokenizer, batch_size=batch_size, max_token_len=max_tok_len)
        data_module.setup()
        baseline = BertBaselineClassifier(self.model_name, self.num_labels, config, self.total_training_steps, self.warmup_steps)

        trainer = pl.Trainer(
            logger=self.logger,
            callbacks=[self.early_stopping_callback],
            max_epochs=2,
            limit_train_batches=0.2,
            limit_val_batches=0.2,
            accelerator='gpu',
            gpus=1
        )

        trainer.fit(baseline, data_module)
        value = trainer.callback_metrics["val_loss"].item()

        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    def get_best_hyperparams(self, n_trials):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        return study.best_params
