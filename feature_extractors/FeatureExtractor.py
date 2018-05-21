from sklearn.base import TransformerMixin, BaseEstimator


class FeatureExtractor(BaseEstimator, TransformerMixin):

  def __init__(self, feature='title'):
    self.feature = feature

  def fit(self, feature_list, y=None):
    return self

  def fit_transform(self, feature_list, y=None):
    return self.transform(feature_list)

  def transform(self, feature_list):
    return [feature for feature in feature_list]

