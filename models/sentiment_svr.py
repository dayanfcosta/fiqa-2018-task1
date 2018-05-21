import timeit
import utilities
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import GridSearchCV
from feature_extractors.Tokenizer import Tokenizer
from sklearn.metrics import mean_squared_error as mse
from feature_extractors.WordReplacement import WordReplacement
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def train(train_sentences, train_sentiments, n_jobs=1, n_cv=10):
  scorer = {'r2': make_scorer(r2, greater_is_better=True),
            'mse': make_scorer(mse, greater_is_better=False)}

  positive_word = ('Excellent word', ['excelent'])
  negative_word = ('Poor word', ['poor'])

  word2vec_model = utilities.word_vector()

  parameters = {
    'tokeniser__ngram_range': [(1, 2, 3)],
    'tokeniser__tokeniser_func': [utilities.tokens],
    'posextract__words_replace': [positive_word],
    'posextract__replacement': ['posword'],
    'posextract__expand': [word2vec_model],
    'posextract__expand_top_n': [10],
    'negextract__words_replace': [negative_word],
    'negextract__replacement': ['negword'],
    'negextract__expand': [word2vec_model],
    'negextract__expand_top_n': [10],
    'count_grams__binary': [True],
    'clf__C': [10, 1, 0.1],
    'count_grams__lowercase': (True, False),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2', None)
  }

  pipeline = Pipeline([
    ('tokeniser', Tokenizer()),
    ('posextract', WordReplacement()),
    ('negextract', WordReplacement()),
    ('count_grams', CountVectorizer(analyzer=utilities.analyzer)),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.LinearSVR())
  ])

  grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=n_cv,
                             scoring=scorer, n_jobs=n_jobs, refit='mse')
  grid_clf = grid_search.fit(train_sentences, train_sentiments)
  return grid_clf
