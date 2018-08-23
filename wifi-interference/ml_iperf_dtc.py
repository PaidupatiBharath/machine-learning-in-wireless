from sklearn.tree import DecisionTreeClassifier as dtc
from time import time
from random_dataset import random_dataset

# Training Model
features_train, labels_train, features_test, labels_test = random_dataset()

model3 = dtc()

t3 = time()
model3.fit(features_train, labels_train)
print "Training time DTC : ", round(time()-t3, 3),"s"


tp3 = time()
predicted = model3.predict(features_test)
print "Predicting time DTC : ", round(time()-tp3, 3),"s"


accuracy3 = model3.score(features_test, labels_test)
print "Accuracy of DTC:", accuracy3
