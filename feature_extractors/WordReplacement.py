from sklearn.base import BaseEstimator, TransformerMixin


class WordReplacement(BaseEstimator, TransformerMixin):

  def __init__(self, words_replace=('name', []), replacement='', expand=None,
               expand_top_n=10, disimlar=('disimlar_name', [])):
    self.words_replace = words_replace
    self.expand_top_n = expand_top_n
    self.replacement = replacement
    self.disimlar = disimlar
    self.expand = expand

  def fit(self, token_list, y=None):
    return self

  def fit_transform(self, token_list, y=None):
    return self.transform(token_list)

  def transform(self, token_list):
    if self.expand:
      expand_replace_words = set()
      for replace_word in self.words_replace[1]:
        expand_replace_words.add(replace_word)
        if replace_word in self.expand.wv.vocab:
          similar_words = self.expand.most_similar(positive=[replace_word],
                                                   negative=self.disimlar[1],
                                                   topn=self.expand_top_n)
          for sim_word, sim_score in similar_words:
            expand_replace_words.add(sim_word)
      self.words_replace = (self.words_replace[0], expand_replace_words)

    all_replace_tokens = []

    for tokens in token_list:
      replace_token_list = []

      for token in tokens:
        splitted_token = token.split()
        replace_tokens = []

        for token_part in splitted_token:
          if token_part in self.words_replace[1]:
            replace_tokens.append(self.replacement)
          else:
            replace_tokens.append(token_part)
        replace_token_list.append(' '.join(replace_tokens))
      all_replace_tokens.append(replace_token_list)

    return all_replace_tokens
