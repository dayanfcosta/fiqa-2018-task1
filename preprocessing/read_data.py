import json
import numpy as np
import utilities


def get_info(id, info_name, data_json):
  return data_json[id]['info'][0][info_name]


def load_data(type, test=False):
  if (type == 'posts'):
    file_path = utilities.TRAIN_POST_FILE_PATH
    if (test):
      file_path = utilities.TEST_POST_FILE_PATH
  else:
    file_path = utilities.TRAIN_HEADLINE_FILE_PATH
    if (test):
      file_path = utilities.TEST_HEADLINE_FILE_PATH

  with open(file_path, 'r', encoding="utf-8") as file:
    ids = []
    aspects = []
    snippets = []
    companies = []
    sentences = []
    sentiment_scores = []

    data_json = json.load(file)

    def getAspects(aspect):
      aspect = aspect.replace('[', '')
      aspect = aspect.replace(']', '')
      aspect = aspect.replace('\'', '')
      return aspect.split('/')

    for id in data_json:
      ids.append(id.lower())
      companies.append(get_info(id, 'target', data_json).lower())
      sentences.append(data_json[id]['sentence'].lower().lower())
      snippets.append(get_info(id, 'snippets', data_json).lower())
      if (not test):
        aspects.append(getAspects(get_info(id, 'aspects', data_json).lower()))
        sentiment_scores.append(float(get_info(id, 'sentiment_score', data_json)))

  if (not test):
    return sentences, snippets, companies, aspects, np.asarray(sentiment_scores)
  return ids, sentences, snippets
