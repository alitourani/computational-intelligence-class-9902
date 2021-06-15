import numpy as np
import pickle

def load_wordvec(word_dict, dictionary_path):
  vectors = {}
  with open(dictionary_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    vectors = dict

  loaded_len = len(vectors)
  print('Loaded', loaded_len, 'already embedded words')

  for word in word_dict:
    if word not in vectors:
      vectors[word] = np.array(np.random.uniform(-5.0, 5.0, 300))
  
  print('Added', len(vectors) - loaded_len, 'new embeddings')
  
  save_wordvec(vectors, dictionary_path)
  print('Embedding saved')

  return vectors

def save_wordvec(vectors, dictionary_path):
  with open(dictionary_path, 'wb') as file:
    pickle.dump(vectors, file, pickle.HIGHEST_PROTOCOL)

def start_embedding():
  vector = {}
  vector['hi'] = np.array(np.random.uniform(-5.0, 5.0, 300))
  save_wordvec(vector, './data/vectors.pkl')
  print ('Hi!')