import numpy as np
from time import time
import cvxopt
from cvxopt import matrix, solvers

test_data = np.loadtxt("test_data.txt")
test_label = np.loadtxt("test_label.txt")

with open('train_data.txt', 'r') as f_in:
    train_data = f_in.readlines()
    train_data = list(map(lambda row: list(map(int, row.split(' '))), train_data))

with open('train_label.txt', 'r') as f_in:
    train_label = f_in.readlines()
    train_label = list(map(lambda row: list(map(int, row.split(' '))), train_label))


third_features = [int(row[2]) for row in train_data]
third_feature_mean = sum(third_features)/1000

def std_dev(data, mean):
    N = len(data)
    return (sum((data[i]-mean)**2 for i in range(N))/(N-1))**0.5

# print(third_feature_mean)
print(std_dev(third_features, third_feature_mean))

tenth_features = [int(row[9]) for row in train_data]
tenth_feature_mean = sum(tenth_features)/1000
# print(tenth_feature_mean)
print(std_dev(tenth_features, tenth_feature_mean))

def normalize(train_data, test_data):
    train_sample_size, feature_size = train_data.shape
    test_sample_size = test_data.shape[0]
    for feat_idx in range(feature_size):
        features = [train_data[i][feat_idx] for i in range(train_sample_size)]
        mean = sum(features)/feature_size
        std_d = std_dev(features, mean)
        for i in range(train_sample_size):
            train_data[i][feat_idx] = (train_data[i][feat_idx] - mean) / std_d
        for i in range(test_sample_size):
            test_data[i][feat_idx] = (test_data[i][feat_idx] - mean) / std_d

    return train_data, test_data

def train_svm(train_data, train_label, C):
    """Train linear SVM (primal form)

    Argument:
      train_data: N*D matrix, each row as a sample and each column as a feature
      train_label: N*1 vector, each row as a label
      C: tradeoff parameter (on slack variable side)

    Return:
      w: feature vector (column vector)
      b: bias term
    """
    w_b_slack_dim = 861
    train_data = np.array(train_data, dtype=np.double)
    train_label = np.array(train_label, dtype=np.double)
    p = np.zeros((861, 861))
    for i in range(60):
        p[i][i] = 1.0

    P = matrix(p)
    
    q = np.zeros(861)
    for i in range(800):
        q[i+61] = C

    Q = matrix(q)

    neg_yx = -np.dot(train_label.T, train_data)
    g = np.zeros((1600, 861))
    
    for i in range(800):
        g[i][:60] = neg_yx
        g[i][60] = train_label[i]

    for i in range(800):
        g[i+800][61+i] = -1.0
        g[i][61+i] = -1.0

    
    G = matrix(g)
    h = matrix(np.array([-1.0]*800 + [0.0]*800).T)
    sol = solvers.qp(P,Q,G,h)
    # print(sol)
    return sol

def test_svm(test_data, test_label, w, b):
    """Test linear SVM

    Argument:
      test_data: M*D matrix, each row as a sample and each column as a feature
      test_label: M*1 vector, each row as a label
      w: feature vector
      b: bias term

    Return:
      test_accuracy: a float between [0, 1] representing the test accuracy
    """
    M, D = test_data.shape
    hit = 0
    w = np.array(w)
    for data, label in zip(test_data, test_label):
        assert(w.shape==(60,1))
        assert(data.shape==(60,))
        data.reshape((60,1))
        pred = np.dot(w.T, data) + b
        if (pred > 0 and label == 1) or (pred < 0 and label == -1):
            hit += 1

    return hit / M

'''
Report the 5-fold cross-validation accuracy (averaged accuracy over each validation set) 
and average training time (averaged over each training subset) on different value of C 
taken from the set {4^−6, 4^−5, · · · , 4^5, 4^6}. '''
train_data = np.array(train_data, dtype=np.double)
train_label = np.array(train_label, dtype=np.double)

train_data, test_data = normalize(train_data, test_data)

def cross_validate(k, total_data, total_label, C):
    avg_accuracy = 0
    avg_time = 0
    batch_size = total_data.shape[0] // k
    assert(batch_size==200)
    for i in range(k):
        start = i*batch_size
        end = start + batch_size
        test_data, test_label = total_data[start:end], total_label[start:end]
        assert(test_data.shape==(200,60))
        assert(test_label.shape==(200,1))
        train_data = np.concatenate((total_data[:start], total_data[end:]))
        assert(train_data.shape==(800,60))
        train_label = np.append(total_label[:start], total_label[end:])
        assert(train_label.shape==(800,))

        t1 = time()
        sol = train_svm(train_data, train_label, C)
        t2 = time()    
        avg_time += t2 - t1
        w = sol['x'][:60]
        b = sol['x'][61]
        # print(len(set(sol['x'][61:])))
        # assert(len(set(sol['x'][61:]))==1)
        accuracy = test_svm(test_data, test_label, w, b)
        avg_accuracy += accuracy

    avg_accuracy /= k
    avg_time /= k
    return avg_accuracy, avg_time

avg_accuracies = []
avg_times = []
C_vals = [4**i for i in range(-6, 7)]
k = 5
from sklearn.model_selection import KFold

for C in C_vals:
    avg_accuracy, avg_time = cross_validate(k, train_data, train_label, C)
    # kf = KFold(n_splits=5)
    # # bs = []
    # for train_index, test_index in kf.split(train_data):
    #     avg_accuracy = 0
    #     avg_time = 0
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = train_data[train_index], test_data[test_index]
    #     y_train, y_test = test_label[train_index], test_label[test_index]
    #     t1 = time()
    #     sol = train_svm(train_data, train_label, C)
    #     t2 = time()    
    #     avg_time += t2 - t1
    #     w = sol['x'][:60]
    #     b = sol['x'][61]
    #     # bs.append(b)
    #     accuracy = test_svm(X_test, y_test, w, b)
    #     print(f'accuracy : {accuracy}')
    #     avg_accuracy += accuracy
    # print('split\n')
    # print(bs)
    avg_accuracies.append(avg_accuracy)
    avg_times.append(avg_time)

print(avg_accuracies)
print(avg_times)