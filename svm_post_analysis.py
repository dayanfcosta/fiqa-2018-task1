from sklearn.preprocessing import MultiLabelBinarizer
from models import aspect_svc as svc
from collections import Counter
from preprocessing import read_data as rd

train_sentences, _, companies, aspects, _ = rd.load_data('posts')
_, test_sentences, _ = rd.load_data('posts', test=True)

## preprocessing aspects
all_aspects = []
for data in aspects:
  for aspect in data:
    all_aspects.append(aspect)

binarizer = MultiLabelBinarizer(classes=list(set(all_aspects)))
train_aspects = binarizer.fit_transform(aspects)

print('Training model...')
grid_clf = svc.train(train_sentences, train_aspects, companies)

print('Model trained, starting predicting...')
clf = grid_clf.best_estimator_
pred_values = clf.predict(test_sentences)

predicted_labels = binarizer.inverse_transform(pred_values)

all_pred = []
for value in predicted_labels:
  for predicted in value:
    all_pred.append(predicted)

print(Counter(all_pred))
print(predicted_labels)
