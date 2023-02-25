import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from utils.data import HumanValuesDataModule
from models.baseline import BertBaselineClassifier

class HyperparameterTuner:
    def __init__(self, data_module, model, study_params):
        
        self.data_module = data_module
        self.model = model

        self.n_trials = study_params["n_trials"]
        self.n_epochs = study_params["n_epochs"]
        self.lim_train_batches = study_params["lim_train_batches"]
        self.lim_val_batches = study_params["lim_val_batches"]
        self.study = None

    def objective(self, trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [8])
        optimizer = trial.suggest_categorical("optim", [torch.optim.Adam, torch.optim.AdamW])
        classifier_dropout = trial.suggest_categorical("optimizer_dropout", [.1, .2, .3])
        max_tok_len = trial.suggest_categorical("max_tok_len", [128, 256, 300, 368])

        self.data_module.batch_size=batch_size
        self.data_module.max_token_len=max_tok_len
        self.data_module.setup()

        self.model.optim = optimizer
        self.model.lr = lr
        self.model.classifier_dropout = classifier_dropout


        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)
        logger = pl.loggers.TensorBoardLogger("lightning_logs", name="human-values")

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback],
            max_epochs=self.n_epochs,
            limit_train_batches=self.lim_train_batches,
            limit_val_batches=self.lim_val_batches,
            accelerator='gpu',
            devices=1
        )

        trainer.fit(self.model, self.data_module)
        value = trainer.callback_metrics["val_loss"].item()

        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    def run_study(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.n_trials)

    def get_best_hyperparams(self):
        if self.study is None:
            raise Exception("Please run the study first.")
        return self.study.best_params
