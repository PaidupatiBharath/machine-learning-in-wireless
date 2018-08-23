from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from time import time
from random_dataset import random_dataset
import numpy as np
#from sklearn.metrics import accuracy_score

# Training Model
features_train, labels_train, features_test, labels_test = random_dataset()

model1 = GaussianNB()
model2 = SVC(kernel='linear')
model3 = dtc()

t1 = time()
model1.fit(features_train, labels_train)
print "Training time NB : ", round(time()-t1, 3),"s"

accuracy1 = model1.score(features_test, labels_test)
print "Accuracy of NB:", accuracy1

t2 = time()
model2.fit(features_train, labels_train)
print "Training time SVM : ", round(time()-t2, 3),"s"

accuracy2 = model2.score(features_test, labels_test)
print "Accuracy of SVM:", accuracy2

t3 = time()
model3.fit(features_train, labels_train)
print "Training time DTC : ", round(time()-t3, 3),"s"

accuracy3 = model3.score(features_test, labels_test)
print "Accuracy of DTC:", accuracy3


pred_data = int(input("Enter the data to be predicted:"))
predicted1 = model1.predict([[pred_data,0]])
print predicted1

predicted2 = model2.predict([[pred_data,0]])
print predicted2

predicted3 = model3.predict([[pred_data,0]])
print predicted3

#val = float(input("Enter the datarate : "))
# tp1 = time()
# predicted1 = model1.predict(features_test)
# print "Predicting time NB : ", round(time()-tp1, 3),"s"

# tp2 = time()
# predicted2 = model2.predict(features_test)
# print "Predicting time SVM : ", round(time()-tp2, 3),"s"

# tp3 = time()
# predicted3 = model3.predict(features_test)
# print "Predicting time DTC : ", round(time()-tp3, 3),"s"



# accuracy = accuracy_score(predicted1, labels_test)
# print "Accuracy for GaussianNB:", accuracy

# accuracy = accuracy_score(predicted2, labels_test)
# print "Accuracy for SVM:", accuracy

# accuracy = accuracy_score(predicted3, labels_test)
# print "Accuracy for DTC:", accuracy
