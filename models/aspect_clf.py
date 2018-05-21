import utilities
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from feature_extractors.Tokenizer import Tokenizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from feature_extractors.WordReplacement import WordReplacement
from feature_extractors.FeatureExtractor import FeatureExtractor
from sklearn.metrics import make_scorer, f1_score, recall_score


def train(train_sentences, train_aspects, companies, n_jobs=1, n_cv=10):
  scorer = {'recall': make_scorer(recall_score, average='micro'),
            'f1': make_scorer(f1_score, average='micro')}

  pos_word = ('Excellent word', ['excellent'])
  neg_word = ('Poor word', ['poor'])

  word2vec_model = utilities.word_vector()

  union_parameters = {
    'union__ngrams__tokeniser__ngram_range': [(1, 2)],
    'union__ngrams__tokeniser__tokeniser_func': [utilities.tokens],
    # 'union__ngrams__text_extract__feature': ['sentence'],
    'union__ngrams__posextract__words_replace': [pos_word],
    'union__ngrams__posextract__replacement': ['posword'],
    'union__ngrams__posextract__expand': [word2vec_model],
    'union__ngrams__posextract__expand_top_n': [10],
    'union__ngrams__negextract__words_replace': [neg_word],
    'union__ngrams__negextract__replacement': ['negword'],
    'union__ngrams__negextract__expand': [word2vec_model],
    'union__ngrams__negextract__expand_top_n': [10],
    'union__ngrams__count_grams__binary': [True],
    # 'union__target_extract__aspect__feature': ['aspects'],
    'union__target_extract__count_grams__binary': [True]
  }

  union_pipeline = Pipeline([
    ('union', FeatureUnion([
      ('ngrams', Pipeline([
        ('text_extract', FeatureExtractor()),
        ('tokeniser', Tokenizer()),
        ('posextract', WordReplacement()),
        ('negextract', WordReplacement()),
        ('count_grams', CountVectorizer(analyzer=utilities.analyzer))
      ])),
      ('target_extract', Pipeline([
        ('aspect', FeatureExtractor()),
        ('count_grams', CountVectorizer(analyzer=utilities.analyzer))
      ])),
    ])),
    ('clf', OneVsRestClassifier(LinearSVC()))
  ])

  grid_search = GridSearchCV(union_pipeline, union_parameters,
                             cv=n_cv, scoring=scorer, n_jobs=n_jobs, refit='f1')
  grid_clf = grid_search.fit(train_sentences, train_aspects)
  return grid_clf
