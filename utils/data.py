import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer
import torch
import pytorch_lightning as pl
import multiprocessing
from torch.utils.data import WeightedRandomSampler

def get_processed_df(input_df, labels_columns, tokenizer_cls_token, tokenizer_sep_token):
    df = pd.DataFrame()

    df['arguments'] = tokenizer_cls_token + input_df['Conclusion'] + tokenizer_sep_token + tokenizer_cls_token + input_df['Stance'] + tokenizer_sep_token + tokenizer_cls_token + input_df['Premise']
    
    df['labels'] = input_df[labels_columns].values.tolist()

    return df


def format_batch_texts(language_code, batch_texts):
  
  formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

  return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr", device='cuda:0'):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True).to(device))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts

def perform_data_augmentation(df_train_aumentation, first_model, second_model, first_model_tkn, second_model_tkn, batch_size=8, columns=['Premise', 'Conclusion'], device='cuda:0'):
    first_model.to(device)
    second_model.to(device)
    for column in columns:
        original_texts = df_train_aumentation[column].to_list()
        dataloader = DataLoader(original_texts, batch_size=batch_size)
        translated_texts = []
        print(f'translating {column}')
        for batch in dataloader:
            translated_batch = perform_translation(batch, first_model, first_model_tkn)
            for text in translated_batch:
                translated_texts.append(text)

        dataloader = DataLoader(translated_texts, batch_size=batch_size)

        back_translated_texts = []
        print(f'back-translating {column}')
        for batch in dataloader:
            back_translated_batch = perform_translation(batch, second_model, second_model_tkn)
            for text in back_translated_batch:
                back_translated_texts.append(text)
        df_train_aumentation[f'{column} BT'] = back_translated_texts
    return df_train_aumentation




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

  def train_dataloader(self):
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