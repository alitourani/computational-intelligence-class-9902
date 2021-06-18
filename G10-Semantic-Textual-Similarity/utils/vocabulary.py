def vocabulary(sentence_pairs):
  vocab_set = set()
  for i in range(len(sentence_pairs)):
    words = sentence_pairs[i][0].split()
    words += sentence_pairs[i][1].split()
    for j in range(len(words)):
      vocab_set.add(words[j])
  vocab = {}
  i = 0
  for word in vocab_set:
    vocab[word] = i
    i += 1

  print('Made a vocabulary of', len(vocab), 'words.')
  return vocab

def add_to_vocabulary(sentence_pairs, input_vocab):
  vocab_set = set()

  for word in input_vocab:
    vocab_set.add(word)

  for i in range(len(sentence_pairs)):
    words = sentence_pairs[i][0].split()
    words += sentence_pairs[i][1].split()
    for j in range(len(words)):
      vocab_set.add(words[j])
  vocab = {}
  i = 0
  for word in vocab_set:
    vocab[word] = i
    i += 1

  print('Made a vocabulary with a total of', len(vocab), 'words.')
  return vocab