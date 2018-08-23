import random
import numpy as np

def random_dataset():
    features = []
    labels = []
    features_test = []
    labels_test = []
    #features_train = []
    
    
    ### ACI Bad
    
    for i in range(1000000):
        a = random.uniform(3,7)
        a= '%.3f'%(a)
        features.append([float(a),0])
        labels.append("ACI BAD")

    ### Test data generation
    
    for i in range(20000):
        a = random.uniform(0,10)
        a = '%.3f'%(a)
        features_test.append([float(a),0])
        labels_test.append("ACI BAD")

    ### ACI Good
    
    # for i in range(1000):
    #     a = random.uniform(23,31)
    #     a= '%.3f'%(a)
    #     b.append(a)

    ### CCI
    
    for i in range(1000000):
        a = random.uniform(14,20)
        a= '%.3f'%(a)
        features.append([float(a), 0])
        labels.append("CCI")

    ### CCI Test data

    for i in range(20000):
        a = random.uniform(11,27)
        a = '%.3f'%(a)
        features_test.append([float(a),0])
        labels_test.append("CCI")


    ### Best case
    
    for i in range(1000000):
        a = random.uniform(35,45)
        a= '%.3f'%(a)
        features.append([float(a), 0])
        labels.append("No Interference")


    for i in range(20000):
        a = random.uniform(28,65)
        a = '%.3f'%(a)
        features_test.append([float(a),0])
        labels_test.append("No Interference")


    features_train = np.array(features)
    labels_train = np.array(labels)
    features_test_transformed = np.array(features_test)
    labels_test_transformed = np.array(labels_test)

    return features_train, labels_train, features_test_transformed, labels_test_transformed    
