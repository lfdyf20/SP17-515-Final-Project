import numpy as np
import scipy.io as io

train_data = np.load('./train_small.npz')
test_data = np.load('./test_small.npz')

train_x = train_data['train_x']
train_labels = train_data['train_labels']

test_x = test_data['test_x']
test_labels = test_data['test_labels']

print('train_x:', type(train_x), train_x.shape)
print('train_labels:', type(train_labels), train_labels.shape)

print('test_x:', type(test_x), test_x.shape)
print('test_labels:', type(test_labels), test_labels.shape)

io.savemat('train.mat',
           dict(train_x=train_x,
                train_labels=train_labels))

io.savemat('test.mat',
           dict(test_x=test_x,
                test_labels=test_labels))
