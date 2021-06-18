import numpy as np
import pickle

def load_wordvec(word_vocab, embed_path, embedding_dim = 300):
  mother_vectors = {}
  with open(embed_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    mother_vectors = dict

  loaded_len = len(mother_vectors)
  print('Loaded', loaded_len, 'already embedded words.')

  current_vectors = {}
  for word in word_vocab:
    if word not in mother_vectors:
      mother_vectors[word] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
    current_vectors[word] = mother_vectors[word]
  
  print('Added', len(mother_vectors) - loaded_len, 'new words to the embedding.')
  print('Loaded', len(current_vectors), 'words to our current dictionary')
  
  save_vec(mother_vectors, embed_path)
  print('Embeddings saved')

  return current_vectors

def reset_word_embedding(embed_path, embedding_dim = 300):
  vector = {}
  vector['hi'] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  save_vec(vector, embed_path)
  print ('Dictionary file created.')

def load_charvec(char_vocab, embed_path, embedding_dim = 100):
  mother_vectors = {}
  with open(embed_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    mother_vectors = dict

  loaded_len = len(mother_vectors)
  print('Loaded', loaded_len, 'already embedded chars')

  current_vectors = {}
  for char in char_vocab:
    if char not in mother_vectors:
      mother_vectors[char] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
    current_vectors[char] = mother_vectors[char]
  
  print('Added', len(mother_vectors) - loaded_len, 'new chars to the embedding.')
  
  save_vec(mother_vectors, embed_path)
  print('Embedding saved.')

  return current_vectors

def reset_char_embedding(embed_path, embedding_dim = 100):
  vector = {}
  vector['a'] = np.array(np.random.uniform(-5.0, 5.0, embedding_dim))
  save_vec(vector, embed_path)
  print('Dictionary file created.')

def save_vec(vectors, embed_path):
  with open(embed_path, 'wb') as file:
    pickle.dump(vectors, file, pickle.HIGHEST_PROTOCOL)

def change_weights_of_mother(vectors, embed_path):
  with open(embed_path, 'rb') as load_file:
    dict = pickle.load(load_file)
    mother_vectors = dict

  for word in vectors:
    if word in mother_vectors:
      mother_vectors[word] = vectors[word]
  
  save_vec(mother_vectors, embed_path)