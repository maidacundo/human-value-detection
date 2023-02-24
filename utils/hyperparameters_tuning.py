import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from utils.data import HumanValuesDataModule
from models.baseline import BertBaselineClassifier

class HyperparameterTuner:
    def __init__(self, data_params, model_params, study_params):
        self.train_df = data_params["train_df"]
        self.val_df = data_params["val_df"]
        self.test_df = data_params["test_df"]
        self.tokenizer = data_params["tokenizer"]

        self.model_name = model_params["model_name"]
        self.num_labels = model_params["num_labels"]
        self.total_training_steps = model_params["total_training_steps"] 
        self.warmup_steps =  model_params["warmup_steps"]

        self.n_trials = study_params["n_trials"]
        self.n_epochs = study_params["n_epochs"]
        self.lim_train_batches = study_params["lim_train_batches"]
        self.lim_val_batches = study_params["lim_val_batches"]
        self.study = None

    def objective(self, trial: optuna.Trial):
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

        baseline = BertBaselineClassifier(self.model_name, self.num_labels, config, self.total_training_steps, self.warmup_steps)

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

        trainer.fit(baseline, data_module)
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
