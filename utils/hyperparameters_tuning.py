import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class HyperparameterTuner:
    def __init__(self, data_module, partial_model, study_params):
        
        self.data_module = data_module
        self.partial_model = partial_model

        self.n_trials = study_params["n_trials"]
        self.n_epochs = study_params["n_epochs"]
        self.lim_train_batches = study_params["lim_train_batches"]
        self.lim_val_batches = study_params["lim_val_batches"]
        
        self.lr_transformer_list = study_params["lr_transformer"]
        self.lr_classifier_list = study_params["lr_classifier"]
        self.weight_decay_list = study_params["weight_decay"]
        self.classifier_dropout_list = study_params["classifier_dropout"]

        self.study = None

    def objective(self, trial: optuna.Trial):
        lr_transformer = trial.suggest_categorical("lr_transformer", self.lr_transformer_list) #[2e-5, 5e-5]
        lr_classifier = trial.suggest_categorical("lr_classifier", self.lr_classifier_list) #[1e-4, 1e-3]
        weight_decay = trial.suggest_categorical("weight_decay", self.weight_decay_list) #[1e-5, 1e-4]
        classifier_dropout = trial.suggest_categorical("classifier_dropout", self.classifier_dropout_list) #[.1, .3]


        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback],
            max_epochs=self.n_epochs,
            limit_train_batches=self.lim_train_batches,
            limit_val_batches=self.lim_val_batches,
            accelerator='gpu'
        )

        model = self.partial_model(classifier_dropout=classifier_dropout, 
                                    optimizer=torch.optim.AdamW, 
                                    lr_transformer=lr_transformer, 
                                    lr_classifier=lr_classifier, 
                                    weight_decay=weight_decay,
                                    )

        trainer.fit(model, self.data_module)
        value = trainer.callback_metrics["val_loss"].item()

        if trial.should_prune():
            raise optuna.TrialPruned()
        
        del model
        torch.cuda.empty_cache()

        return value

    def run_study(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.n_trials, gc_after_trial=True)

    def get_best_hyperparams(self):
        if self.study is None:
            raise Exception("Please run the study first.")
        return self.study.best_params


{'lr_transformer': 5e-05, 'lr_classifier': 0.001, 'weight_decay': 1e-05, 'classifier_dropout': 0.1}