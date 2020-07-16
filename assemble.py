import numpy as np
import csv

bert = np.load('bert.npy', allow_pickle=True)
lstm = np.load('lstm.npy', allow_pickle=True)
gru = np.load('gru.npy', allow_pickle=True)

labels = np.load('labels.npy', allow_pickle=True)
test_ids = np.load('test_ids.npy', allow_pickle=True)

results = []

TOTAL_TEST_CASES = 6702
TOTAL_CATEGORIES = 352

if len(labels) != TOTAL_CATEGORIES:
  print('Labels file is incorrect')

if len(test_ids) != TOTAL_TEST_CASES:
  print('Test ids file is incorrect')

def assertShape(title, data):
  try:
    assert data.shape[0] == 6702
    assert data.shape[1] == TOTAL_CATEGORIES
    print(title, ' loaded successfuly')
  except:
    print('Error loading ', title, ' shape is incorrect')

assertShape('LSTM Model', lstm)
assertShape('GRU Model', gru)
assertShape('BERT Model', bert)

print('...working.... ')

# For each item calc the category
for i, item in enumerate(test_ids):
  # Sum over all categories with the selected weights
  scores = [(bert[i][k]*0.5 + lstm[i][k]*0.3 + gru[i][k]*0.2) for k in range(len(labels))]

  # Append as result the winning label
  results.append(labels[np.argmax(scores)].replace('Cat_', ''))

# Write the file
with open('submission.csv', mode='w', newline='') as base_file:
  csv_iterator = csv.writer(base_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for i, value in enumerate(results):
    csv_iterator.writerow([test_ids[i], value])

print('** submission.csv generated **')

    