import numpy as np
def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(':')[0])
                ifidf = float(index_tfidf.split(':')[1])
                r_d[index] = ifidf
            return np.array(r_d)
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())
    X = []
    y = []
    for data_id , d in enumerate(d_lines):
        features = d.split('<fff>')
        label = int(features[0])
        r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
        X.append(r_d)
        y.append(label)
    return X, np.array(y)
def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / expected_y.size
    return accuracy
train_X, train_y = load_data(data_path = 'D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/20news-train-tfidf.txt')
from sklearn.svm import SVC
classifier_kernel = SVC(
    C = 50.0,    
    kernel = "rbf",       
    gamma = 0.1,
    tol = 0.001,        
    verbose = True    
)
classifier_kernel.fit(train_X, train_y)

test_X, test_y = load_data(data_path = 'D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/20news-test-tfidf.txt')
predicted_y = classifier_kernel.predict(test_X)
accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
print("Accuracy: {}".format(accuracy))



