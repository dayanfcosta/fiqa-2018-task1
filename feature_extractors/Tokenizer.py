import utilities
from sklearn.base import BaseEstimator, TransformerMixin


class Tokenizer(BaseEstimator, TransformerMixin):

  def __init__(self, ngram_range=(1, 1), tokeniser_func=utilities.tokens):
    self.ngram_range = ngram_range
    self.tokeniser_func = tokeniser_func

  def fit(self, texts, y=None):
    return self

  def fit_transform(self, texts, y=None):
    return self.transform(texts)

  def transform(self, texts):
    tokens = [self.tokeniser_func(text) for text in texts]
    n_gram_tokens = utilities.ngrams(tokens, self.ngram_range)
    return n_gram_tokens
