from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MultiLabelBinarizer
from models import aspect_svc as svc

from preprocessing import read_data as rd

sentences, _, companies, aspects, _ = rd.load_data('headlines')
train_sentences, test_sentences, train_aspects, test_aspects = tts(sentences, aspects, test_size=0.2)

## preprocessing aspects
all_aspects = []
for data in aspects:
  for aspect in data:
    all_aspects.append(aspect)

print(all_aspects)

binarizer = MultiLabelBinarizer(classes=list(set(all_aspects)))
Y_train = binarizer.fit_transform(train_aspects)
Y_test = binarizer.fit_transform(test_aspects)

print('Training model...')
grid_clf = svc.train(train_sentences, Y_train, companies)

print('Model trained, starting predicting...')
clf = grid_clf.best_estimator_
pred_values = clf.predict(test_sentences)

from sklearn.metrics import f1_score, precision_score, recall_score

print('Values predicted!')
f1 = f1_score(Y_test, pred_values, average='weighted')
recall = recall_score(Y_test, pred_values, average='weighted')
precision = precision_score(Y_test, pred_values, average='weighted')
print('HeadLine Precision: ', precision)
print('HeadLine Recall: ', recall)
print('HeadLine F1-Score: ', f1)
