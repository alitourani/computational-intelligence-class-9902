import pandas as pd

class Dataset():
  def __init__(self, data_path, test_ratio = 0.1):
    dataset = pd.read_csv(
      data_path,
      delimiter = '\t',
    )
    dataset.dropna(inplace=True)
    dataset = dataset.drop(columns = ['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
    'captionID', 'pairID'])
    dataset['score'] = (0.5 + 
      ((((dataset['label1'] == 'entailment').astype(int) + (dataset['label2'] == 'entailment').astype(int) + 
      (dataset['label3'] == 'entailment').astype(int) + (dataset['label4'] == 'entailment').astype(int) + 
      (dataset['label5'] == 'entailment').astype(int)) * 0.1) - (((dataset['label1'] == 'contradiction').astype(int) + 
      (dataset['label2'] == 'contradiction').astype(int) + (dataset['label3'] == 'contradiction').astype(int) + 
      (dataset['label4'] == 'contradiction').astype(int) + (dataset['label5'] == 'contradiction').astype(int)) * 0.1)))
    # dataset['score'] = (0.5 + 
    #   ((dataset['label1'] == 'entailment').astype(int) / 10) - ((dataset['label1'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label2'] == 'entailment').astype(int) / 10) - ((dataset['label2'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label3'] == 'entailment').astype(int) / 10) - ((dataset['label3'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label4'] == 'entailment').astype(int) / 10) - ((dataset['label4'] == 'contradiction').astype(int) / 10) +
    #   ((dataset['label5'] == 'entailment').astype(int) / 10) - ((dataset['label5'] == 'contradiction').astype(int) / 10))

    dataset = dataset.drop(columns = ['label1', 'label2', 'label3', 'label4', 'label5'])
    dataset = dataset.sample(frac = 1, random_state = 1).reset_index(drop = True)
    num_instances = len(dataset)
    self.num_train = num_instances * (1 - test_ratio) 
    self.num_test = num_instances * test_ratio
    self.train_dataset = dataset.loc[:self.num_train]
    self.test_dataset = dataset.loc[self.num_train : self.num_train + self.num_test]

  def train_set(self):
    return self.train_dataset

  def train_set_pairs(self):
    return self.train_dataset[['sentence1', 'sentence2']].values

  def train_set_scores(self):
    return self.train_dataset['score'].values

  def train_set_num(self):
    return len(self.train_dataset)

  def test_set(self):
    return self.test_dataset

  def test_set_pairs(self):
    return self.test_dataset[['sentence1', 'sentence2']].values

  def test_set_scores(self):
    return self.test_dataset['score'].values

  def test_set_num(self):
    return len(self.test_dataset)