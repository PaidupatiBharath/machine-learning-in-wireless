from sklearn.naive_bayes import GaussianNB
from time import time
from random_dataset import random_dataset

# Training Model
features_train, labels_train, features_test, labels_test = random_dataset()

model1 = GaussianNB()

t1 = time()
model1.fit(features_train, labels_train)
print "Training time NB : ", round(time()-t1, 3),"s"


#val = float(input("Enter the datarate : "))
tp1 = time()
predicted = model1.predict(features_test)
print "Predicting time NB : ", round(time()-tp1, 3),"s"



accuracy1 = model1.score(features_test, labels_test)
print "Accuracy of NB:", accuracy1

