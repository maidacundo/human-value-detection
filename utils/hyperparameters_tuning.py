import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from utils.data import HumanValuesDataModule
from models.baseline import BertBaselineClassifier

class HyperparameterTuner:
    def __init__(self, train_df, val_df, test_df, tokenizer, model_name, num_labels,
                 total_training_steps, warmup_steps):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_labels = num_labels
        self.total_training_steps = total_training_steps
        self.warmup_steps = warmup_steps
        self.study = None

    def objective(self, trial: optuna.Trial, n_epochs=2, lim_train_batches=0.2, lim_val_batches=0.2):
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

        data_module = HumanValuesDataModule(
            self.train_df, self.val_df, self.test_df, self.tokenizer, batch_size=batch_size, max_token_len=max_tok_len)
        data_module.setup()

        baseline = BertBaselineClassifier(self.model_name, self.num_labels, config)

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)
        logger = pl.loggers.TensorBoardLogger("lightning_logs", name="human-values")

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback],
            max_epochs=n_epochs,
            limit_train_batches=lim_train_batches,
            limit_val_batches=lim_val_batches,
            accelerator='gpu',
            devices=1
        )

        trainer.fit(baseline, data_module)
        value = trainer.callback_metrics["val_loss"].item()

        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    def run_study(self, n_trials=10):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=n_trials)

    def get_best_hyperparams(self):
        if self.study is None:
            raise Exception("Please run the study first.")
        return self.study.best_params
