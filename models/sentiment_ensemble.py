import utilities
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from models import sentiment_helper as helper
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import GridSearchCV
from feature_extractors.Tokenizer import Tokenizer
from feature_extractors.WordReplacement import WordReplacement
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

##
# This file was used just to test an ensemble technique.
# In this case we tested Stacking Regressor but using just LinearSVR was better
##
def train(train_sentences, train_sentiments, n_jobs=1, n_cv=10):
  forest = RandomForestRegressor(n_estimators=100)
  regressors = (LinearSVR(), Lasso(), AdaBoostRegressor(), BaggingRegressor(), GradientBoostingRegressor())
  stack = StackingRegressor(regressors=regressors, meta_regressor=forest)

  params = {'model__linearsvr__C': [10, 1, 0.1, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06]}

  scorer, parameters = helper.scorer_params(params)

  pipeline = Pipeline([
    ('tokeniser', Tokenizer()),
    ('posextract', WordReplacement()),
    ('negextract', WordReplacement()),
    ('count_grams', CountVectorizer(analyzer=utilities.analyzer)),
    ('tfidf', TfidfTransformer()),
    ('model', stack)
  ])

  grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=n_cv,
                             scoring=scorer, n_jobs=n_jobs, refit='mse', verbose=2)
  grid_clf = grid_search.fit(train_sentences, train_sentiments)
  return grid_clf
