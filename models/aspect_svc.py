from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn import ensemble
from sklearn.metrics import make_scorer

import utilities
from feature_extractors.FeatureExtractor import FeatureExtractor
from feature_extractors.ToList import ToList
from feature_extractors.Tokenizer import Tokenizer
from feature_extractors.WordReplacement import WordReplacement


def scorers(actual, predicted):
  fscore = f1_score(actual, predicted, average='macro')
  precision = precision_score(actual, predicted, average='macro')
  recall = recall_score(actual, predicted, average='macro')
  return (fscore + precision + recall) / 3


def train(train_data, train_aspects, companies, n_jobs=1, n_cv=10):
  scorer = make_scorer(scorers)

  pos_word = ('Excellent word', ['excellent'])
  neg_word = ('Poor word', ['poor'])

  word2vec_model = utilities.word_vector()
  union_parameters = {
    'union__ngrams__tokeniser__ngram_range': [(1, 2, 3)],
    'union__ngrams__tokeniser__tokeniser_func': [utilities.tokens],
    'union__ngrams__text_extract__feature': ['sentence'],
    # 'union__ngrams__compextract__words_replace': [companies],
    # 'union__ngrams__compextract__replacement': ['companyname'],
    # 'union__ngrams__compextract__expand': [None],
    'union__ngrams__posextract__words_replace': [pos_word],
    'union__ngrams__posextract__replacement': ['posword'],
    'union__ngrams__posextract__expand': [word2vec_model],
    'union__ngrams__posextract__expand_top_n': [10],
    'union__ngrams__negextract__words_replace': [neg_word],
    'union__ngrams__negextract__replacement': ['negword'],
    'union__ngrams__negextract__expand': [word2vec_model],
    'union__ngrams__negextract__expand_top_n': [10],
    'union__ngrams__count_grams__binary': [True],
    'union__ngrams__count_grams__lowercase': (True, False),
    'union__ngrams__tfidf__use_idf': (True, False),
    'union__ngrams__tfidf__norm': ('l1', 'l2'),
    'union__target_extract__aspect__feature': ['aspect'],
    'union__target_extract__count_grams__binary': [True],
    'clf__estimator__C': [10, 1, 0.1]
    # 'clf__estimator__n_estimators': [10, 20, 30, 40, 50],
    # 'clf__estimator__max_depth': [5, 10, 20, 30]
  }

  union_pipeline = Pipeline([
    ('union', FeatureUnion([
      ('ngrams', Pipeline([
        ('text_extract', FeatureExtractor()),
        ('tokeniser', Tokenizer()),
        # ('compextract', WordReplacement()),
        ('posextract', WordReplacement()),
        ('negextract', WordReplacement()),
        ('count_grams', CountVectorizer(analyzer=utilities.analyzer)),
        ('tfidf', TfidfTransformer()),
      ])),
      ('target_extract', Pipeline([
        ('aspect', FeatureExtractor()),
        ('aspect_list', ToList()),
        ('count_grams', CountVectorizer(analyzer=utilities.analyzer))
      ])),
    ])),
    ('clf', OneVsRestClassifier(LinearSVC()))
  ])

  grid_search = GridSearchCV(union_pipeline, param_grid=union_parameters,
                             cv=n_cv, n_jobs=n_jobs, scoring=scorer)
  grid_clf = grid_search.fit(train_data, train_aspects)
  return grid_clf
