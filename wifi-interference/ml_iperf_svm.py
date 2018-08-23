from sklearn.svm import SVC
from time import time
from random_dataset import random_dataset

# Training Model
features_train, labels_train, features_test, labels_test = random_dataset()


model2 = SVC(kernel='rbf')



t2 = time()
model2.fit(features_train, labels_train)
print "Training time SVM : ", round(time()-t2, 3),"s"


#val = float(input("Enter the datarate : "))

tp2 = time()
predicted = model2.predict(features_test)
print "Predicting time SVM : ", round(time()-tp2, 3),"s"




accuracy2 = model2.score(features_test, labels_test)
print "Accuracy of SVM:", accuracy2

