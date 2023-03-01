import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer
import torch
import pytorch_lightning as pl
import multiprocessing
from torch.utils.data import WeightedRandomSampler

def get_processed_df(input_df, labels_columns, tokenizer_sep_token):
    df = pd.DataFrame()

    df['arguments'] = input_df['Conclusion'] + tokenizer_sep_token + input_df['Stance'] + tokenizer_sep_token + input_df['Premise']
    df['labels'] = input_df[labels_columns].values.tolist()

    return df

class HumanValuesDataset(Dataset):

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: BertTokenizer, 
    max_token_len,
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    arguments = data_row['arguments']
    labels = data_row['labels']

    encoding = self.tokenizer(
      arguments,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      arguments=arguments,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )

class HumanValuesDataModule(pl.LightningDataModule):

  def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.num_workers = multiprocessing.cpu_count()

  def setup(self, stage=None):
    self.train_dataset = HumanValuesDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.val_dataset = HumanValuesDataset(
      self.val_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = HumanValuesDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )
  # Calculate class weights for each label
    self.class_weights = []
    for label_idx in range(len(self.train_dataset[0]['labels'])):
        class_count = [0, 0]
        for data in self.train_dataset:
            label = data['labels'][label_idx]
            class_count[int(label)] += 1
        self.class_weights.append(sum(class_count) / (2 * class_count[1] + class_count[0]))

  def train_dataloader(self):
    if self.oversampling:
        # Use weighted sampler for oversampling minority classes
        weights = [self.class_weights[int(label_idx)] for label_idx in range(len(self.train_dataset[0]['labels']))]
        sampler = WeightedRandomSampler(weights, len(self.train_dataset))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers
        )
    else:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )


  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

  def threshold_train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )