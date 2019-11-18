import pickle

import plot_util

results = None
with open('../logs/1571429342', 'rb') as in_file:
    results = pickle.load(in_file)

print(results.keys())
print(plot_util.plot_train_penalty(results))
print(plot_util.plot_raw_train_loss(results))
print(plot_util.plot_train_acc(results))
print(plot_util.plot_tune_acc(results))
