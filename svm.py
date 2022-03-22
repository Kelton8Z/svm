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

train_data = np.array(train_data, dtype=np.double)
train_label = np.array(train_label, dtype=np.double)

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
    sample_size, feature_size = train_data.shape
    w_b_slack_dim = 2*sample_size+feature_size
    p = np.zeros((w_b_slack_dim, w_b_slack_dim))
    for i in range(sample_size):
        p[i][i] = 1.0
    # P = matrix(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.double))
    # w (60,)
    # b (1000,)
    # slack (1000,)
    # -> P (2060, 2060)
    P = matrix(p)
    q = np.zeros(w_b_slack_dim)
    for i in range(sample_size):
        q[i+sample_size+feature_size] = C
    # Q = matrix(np.array([0, 0, C]))
    Q = matrix(q)
    # y/train_label is (1000,)
    # x/train_data is (1000, 60)
    
    
    g_top = np.append(-np.dot(train_data.T, train_label), np.append(-train_label.T, np.array([-1.0]*sample_size)))
    g_bottom = np.array([0.0]*(feature_size + sample_size) + [-1.0]*sample_size)

    # g = np.array([g_top, g_bottom], dtype=np.double)
    g = np.append(g_top, g_bottom, axis=1)
    print(g.shape)
    # G (2, 2060)
    G = matrix(g)
    h = np.zeros(w_b_slack_dim)
    h[:feature_size+sample_size] = -1.0
    # H = matrix(np.array([-1.0, 0.0]).T)
    H = matrix(h)
    sol = solvers.qp(P,Q,G,H)
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
    for data, label, b_val in zip(test_data, test_label, b):
        # data = data.reshape((60,1))
        pred = np.dot(w.T, data) + b_val 
        if (pred > 0 and label == 1) or (pred < 0 and label == -1):
            hit += 1

    return hit / M

'''
Report the 5-fold cross-validation accuracy (averaged accuracy over each validation set) 
and average training time (averaged over each training subset) on different value of C 
taken from the set {4^−6, 4^−5, · · · , 4^5, 4^6}. '''
def cross_validate(k, total_data, total_label, C):
    avg_accuracy = 0
    avg_time = 0
    batch_size = total_data.shape[0] // k
    for i in range(k):
        start = i*batch_size
        end = start + batch_size
        test_data, test_label = total_data[start:end], total_label[start:end]
        train_data = np.concatenate((total_data[:start], total_data[end:]))
        train_label = np.append(total_label[:start], total_label[end:])

        t1 = time()
        sol = train_svm(train_data, train_label, C)
        t2 = time()    
        avg_time += t2 - t1
        w = sol['x'][:60]
        b = sol['x'][60:1060]
        accuracy = test_svm(test_data, test_label, w, b)
        avg_accuracy += accuracy

    avg_accuracy /= 5
    avg_time /= 5
    return avg_accuracy, avg_time

C_vals = [4**i for i in range(-6, 7)]
avg_accuracies = []
avg_times = []
# (train_data.shape) (1000, 60)
# (test_data.shape) (2175, 60)

train_data, test_data = normalize(train_data, test_data)
total_data = np.concatenate((train_data, test_data))
total_label = np.append(train_label, test_label)

k = 5
for C in C_vals:
    avg_accuracy, avg_time = cross_validate(k, total_data, total_label, C)
    avg_accuracies.append(avg_accuracy)
    avg_times.append(avg_time)

print(avg_accuracies)
print(avg_times)