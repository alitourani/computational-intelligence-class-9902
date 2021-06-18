def vocabulary(sentence_pairs):
  vocab = set()
  for i in range(len(sentence_pairs)):
    words = sentence_pairs[i][0].split()
    words += sentence_pairs[i][1].split()
    for j in range(len(words)):
      vocab.add(words[j])

  return vocab