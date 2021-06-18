import numpy as np
import pickle

def load_wordvec(word_dict, dictionary_path, embedding_dim = 300):
  vectors = {}
  with open(dictionary_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    vectors = dict

  loaded_len = len(vectors)
  print('Loaded', loaded_len, 'already embedded words.')

  for word in word_dict:
    if word not in vectors:
      vectors[word] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  
  print('Added', len(vectors) - loaded_len, 'new words to the embedding.')
  
  save_vec(vectors, dictionary_path)
  print('Embedding saved')

  return vectors

def start_word_embedding(path, embedding_dim = 300):
  vector = {}
  vector['hi'] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  save_vec(vector, path)
  print ('Dictionary file created.')

def load_charvec(char_dict, dictionary_path, embedding_dim = 100):
  vectors = {}
  with open(dictionary_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    vectors = dict

  loaded_len = len(vectors)
  print('Loaded', loaded_len, 'already embedded chars')

  for char in char_dict:
    if char not in vectors:
      vectors[char] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  
  print('Added', len(vectors) - loaded_len, 'new chars to the embedding.')
  
  save_vec(vectors, dictionary_path)
  print('Embedding saved.')

  return vectors

def start_char_embedding(path, embedding_dim = 100):
  vector = {}
  vector['a'] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  save_vec(vector, path)
  print('Dictionary file created.')

def save_vec(vectors, dictionary_path):
  with open(dictionary_path, 'wb') as file:
    pickle.dump(vectors, file, pickle.HIGHEST_PROTOCOL)