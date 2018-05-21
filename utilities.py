import json
import re
import warnings

import gensim
import numpy as np
import unitok.configs.english
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from unitok import unitok as tokenizer

WORD2VEC_MODEL_PATH = './models/word2vec_model/all_fin_model_lower'

TRAIN_HEADLINE_FILE_PATH = './datasets/task1_headline_ABSA_train.json'
TEST_HEADLINE_FILE_PATH = './datasets/task1_headline_ABSA_test.json'
TRAIN_POST_FILE_PATH = './datasets/task1_post_ABSA_train.json'
TEST_POST_FILE_PATH = './datasets/task1_post_ABSA_test.json'

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def word_vector():
  return gensim.models.Word2Vec.load(WORD2VEC_MODEL_PATH)


## TOKENIZER
def tokens(text):
  tokens = tokenizer.tokenize(text, unitok.configs.english)
  return [token for tag, token in tokens if token.strip() and tag != 'URL']


def analyzer(token):
  return token


def ngrams(token_list, n_range):
  def get_n_grams(temp_tokens, n):
    token_copy = list(temp_tokens)
    gram_tokens = []
    while (len(token_copy) >= n):
      n_list = []
      for i in range(0, n):
        n_list.append(token_copy[i])
      token_copy.pop(0)
      gram_tokens.append(' '.join(n_list))
    return gram_tokens

  all_n_grams = []
  for tokens in token_list:
    if n_range == (1, 1):
      all_n_grams.append(tokens)
    else:
      all_tokens = []
      for n in range(n_range[0], n_range[1] + 1):
        all_tokens.extend(get_n_grams(tokens, n))
      all_n_grams.append(all_tokens)

  return all_n_grams


def stats_report(clf, file_name):
  def convert_value(value):
    if callable(value):
      value = value.__name__
    return str(value)

  means_r2 = clf.cv_results_['mean_test_r2']
  means_mse = clf.cv_results_['mean_test_mse']
  stds_mse = clf.cv_results_['std_test_mse']
  stds_r2 = clf.cv_results_['std_test_r2']
  params = clf.cv_results_['params']

  with open(file_name, 'w') as file:
    file.write("mean_r2;std_r2;mean_mse;std_mse;{} \n".format(';'.join(params[0].keys())))
    for mean_r2, mean_mse, std_r2, std_mse, param in zip(means_r2, means_mse, stds_r2, stds_mse, params):
      param_values = []
      for key, value in param.items():
        if ('__words_replace' in key or '__disimlar' in key or
            '__word2extract' in key):
          param_values.append(convert_value(value[0]))
        else:
          param_values.append(convert_value(value))
      file.write("{};{};{};{};{}\n".format(str(mean_r2), str(std_r2),
                                           str(mean_mse), str(std_mse),
                                           ';'.join(param_values)))


def eval_sentiment_format(sentence_list, sentiment_list):
  assert len(sentence_list) == len(sentiment_list), 'The two list have to be of the same length'

  return [{'sentence': sentence_list[i], 'sentiment_score': sentiment_list[i]} for
          i in range(len(sentence_list))]


def eval_aspect_format(sentences, aspects):
  assert len(sentences) == len(aspects), 'The two list have to be of the same length'

  return [{'sentence': sentences[i], 'aspects': aspects[i]} for
          i in range(len(sentences))]


def eval_aspects(true_values, predicted_values, metric):
  sentence_id = {}
  true_aspects = []
  predicted_aspects = []

  for i in range(len(true_values)):
    data = true_values[i]
    ids = sentence_id.get(data['sentence'], [])
    ids.append(i)
    sentence_id[data['sentence']] = ids
    true_aspects.append(true_values[i]['aspects'])
    predicted_aspects.append(predicted_values[i]['aspects'])

    return metric(true_aspects, predicted_aspects)


def eval_func(true_values, predicted_values, metric):
  sentence_id = {}
  true_sentiments = []
  predicted_sentiments = []

  for i in range(len(true_values)):
    data = true_values[i]
    ids = sentence_id.get(data['sentence'], [])
    ids.append(i)
    sentence_id[data['sentence']] = ids
    true_sentiments.append(true_values[i]['sentiment_score'])
    predicted_sentiments.append(predicted_values[i]['sentiment_score'])

  return metric(true_sentiments, predicted_sentiments)


def eval_aspect_func(true_values, predicted_values, metric):
  sentence_id = {}
  true_sentiments = []
  predicted_aspects = []

  for i in range(len(true_values)):
    data = true_values[i]
    ids = sentence_id.get(data['sentence'], [])
    ids.append(i)
    sentence_id[data['sentence']] = ids
    true_sentiments.append(true_values[i]['aspects'])
    predicted_aspects.append(predicted_values[i]['aspects'])

  return metric(true_sentiments, predicted_aspects)


def pred_true_diff(pred_values, true_values, score_function, mapping=None):
  results = []

  for i in range(len(pred_values)):
    mapped_value = i
    if hasattr(mapping, '__index__') or hasattr(mapping, 'index'):
      mapped_value = mapping[i]
    results.append((mapped_value, pred_values[i], score_function([pred_values[i]], [true_values[i]])))


def error_cross_validate(train_data, train_values, model, n_folds=10, shuffle=True,
                         score_function=mean_absolute_error):
  results = []
  train_data_array = np.asarray(train_data)
  train_values_array = np.asarray(train_values)

  kfold = KFold(n_splits=n_folds, shuffle=shuffle)
  for train, test in kfold.split(train_data_array, train_values_array):
    model.fit(train_data_array[train], train_values_array[train])
    predicted_values = model.predict(train_data_array[test])
    real_values = train_values_array[test]
    results.extend(pred_true_diff(predicted_values, real_values, score_function, mapping=test))
  return results


def top_n_errors(error_res, train_data, train_values, n=10):
  error_res = sorted(error_res, key=lambda value: value[2], reverse=True)
  top_errors = error_res[:n]
  return [{'Sentence: ': train_data[index],
           'True value: ': train_values[index],
           'Predicted values: ': pred_value,
           'Index: ': index} for index, pred_value, _ in top_errors]


def error_analysis(data, values, clf, cv=None, num_errors=50, score_function=mean_absolute_error):
  if cv:
    if isinstance(cv, dict):
      error_results = error_cross_validate(data, values, clf, score_function=score_function, **cv)
    else:
      error_results = error_cross_validate(data, values, clf, score_function=score_function)
  else:
    pred_values = clf.predict(data)
    error_results = pred_true_diff(pred_values, values, score_function)

  top_errors = top_n_errors(error_results, data, values, n=num_errors)


def replace_url(content):
  return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', content)


def evaluation_format(ids, snippets, sentiments, aspects, file_name):
  data = {
    'team': 'inf-ufg',
    'paper': 'INF-UFG at FiQA 2018 Task 1: Predicting Sentiments and Aspects on Financial Tweets and News Headlines'
  }
  results = []
  for i in range(len(ids)):
    aspect_separator = '/'
    aspect = aspect_separator.join(re.sub('[^a-zA-Z0-9 \n\.]', '', str(a)) for a in aspects[i])
    result = {
      'id': ids[i],
      'snippet': snippets[i],
      'aspect_categories': aspect,
      'sentiment_score': sentiments[i]
    }
    results.append(result)
    data['results'] = results
  with open(file_name + '.json', 'w') as json_file:
    json.dump(data, json_file)
