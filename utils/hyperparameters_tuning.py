import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

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
        lr_transformer = trial.suggest_categorical("lr_transformer", [1e-5, 2e-5, 5e-5])
        lr_classifier = trial.suggest_categorical("lr_classifier", [1e-5, 1e-4, 1e-3])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
        optimizer = torch.optim.AdamW
        classifier_dropout = trial.suggest_categorical("classifier_dropout", [.1, .2, .3])

        self.model.optim = optimizer
        self.model.lr_transformer = lr_transformer
        self.model.lr_classifier = lr_classifier
        self.model.classifier_dropout = classifier_dropout
        self.model.weight_decay = weight_decay


        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)

        trainer = pl.Trainer(
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
