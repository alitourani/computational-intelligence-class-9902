import pandas as pd

class Dataset():
  def __init__(self, data_path, config, validation_ratio = 0.2, test_ratio = 0.2):
    dataset = pd.read_csv(
      data_path,
      delimiter = '\t',
    )
    self.loss_mode = config['model']['loss_mode']
    dataset.dropna(inplace=True)
    dataset = dataset.drop(columns = ['sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
    'captionID', 'pairID'])
    if self.loss_mode == 'mse':
      dataset['score'] = (0.5 + 
        ((((dataset['label1'] == 'entailment').astype(int) + (dataset['label2'] == 'entailment').astype(int) + 
        (dataset['label3'] == 'entailment').astype(int) + (dataset['label4'] == 'entailment').astype(int) + 
        (dataset['label5'] == 'entailment').astype(int)) * 0.1) - (((dataset['label1'] == 'contradiction').astype(int) + 
        (dataset['label2'] == 'contradiction').astype(int) + (dataset['label3'] == 'contradiction').astype(int) + 
        (dataset['label4'] == 'contradiction').astype(int) + (dataset['label5'] == 'contradiction').astype(int)) * 0.1)))
    elif self.loss_mode == 'cel':
      dataset['score'] = (0 + ((dataset['gold_label'] == 'neutral').astype(int) * 1) + ((dataset['gold_label'] == 'entailment').astype(int) * 2))
    # dataset['score'] = (0.5 + 
    #   ((dataset['label1'] == 'entailment').astype(int) / 10) - ((dataset['label1'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label2'] == 'entailment').astype(int) / 10) - ((dataset['label2'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label3'] == 'entailment').astype(int) / 10) - ((dataset['label3'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label4'] == 'entailment').astype(int) / 10) - ((dataset['label4'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label5'] == 'entailment').astype(int) / 10) - ((dataset['label5'] == 'contradiction').astype(int) / 10))

    dataset = dataset.drop(columns = ['gold_label', 'label1', 'label2', 'label3', 'label4', 'label5'])
    dataset = dataset.sample(frac = 1, random_state = 1).reset_index(drop = True)
    num_instances = len(dataset)
    self.num_train = num_instances * (1 - test_ratio - validation_ratio) 
    self.num_test = num_instances * test_ratio
    self.num_validation = num_instances * validation_ratio
    self.validation_dataset = dataset.loc[:self.num_validation]
    self.test_dataset = dataset.loc[self.num_validation : self.num_validation + self.num_test]
    self.train_dataset = dataset.loc[self.num_validation + self.num_test : self.num_train + self.num_validation + self.num_test]

  def train_set(self):
    return self.train_dataset

  def train_set_pairs(self):
    return self.train_dataset[['sentence1', 'sentence2']].values

  def train_set_scores(self):
    return self.train_dataset['score'].values

  def train_set_num(self):
    return len(self.train_dataset)

  def validation_set(self):
    return self.validation_dataset
  
  def validation_set_pairs(self):
    return self.validation_dataset[['sentence1', 'sentence2']].values

  def validation_set_scores(self):
    return self.validation_dataset['score'].values

  def validation_set_num(self):
    return len(self.validation_dataset)

  def test_set(self):
    return self.test_dataset

  def test_set_pairs(self):
    return self.test_dataset[['sentence1', 'sentence2']].values

  def test_set_scores(self):
    return self.test_dataset['score'].values

  def test_set_num(self):
    return len(self.test_dataset)