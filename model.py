import numpy as np 
import pandas as pd
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve, plot_precision_recall_curve
from keras.wrappers.scikit_learn import KerasClassifier
from google.colab import files

#renaming real and fake directories
real = "/content/drive/My Drive/archive/real_and_fake_face/training_real"
fake = "/content/drive/My Drive/archive/real_and_fake_face/training_fake"
#we're creating a list of real and fake images
real_path = os.listdir(real)
fake_path = os.listdir(fake)
shuffle(fake_path)

#Preprossesing the dataset; resize, grayscale, adding label
#Splitting 80:20 into training and test
img_size = 128
def create_training_data():
    training_data = []
    for img in tqdm(real_path[:865]):
        path = os.path.join(real, img)
        label = [1] 
        image = cv2.resize( cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size) )
        training_data.append([np.array(image), np.array(label)])
        
    for img in tqdm(fake_path[:768]):
        path = os.path.join(fake, img)
        label = [0] 
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size))
        training_data.append([np.array(image), np.array(label)])
    return(training_data)

#Creating a list of test data
def create_test_data():
    test_data = []
    for img in tqdm(real_path[865:]):
        path = os.path.join(real, img)
        label = [1] 
        image = cv2.resize( cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size) )
        test_data.append([np.array(image), np.array(label)])
        
    for img in tqdm(fake_path[768:]):
        path = os.path.join(fake, img)
        label = [0] 
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size))
        test_data.append([np.array(image), np.array(label)])
    return(test_data)  
train_data = create_training_data()
test_data = create_test_data()

#Preprocessing 
X = []
y = []

for i in train_data:
    X.append(i[0])
    y.append(i[1])
    
#print(X[0].reshape(-1, 50, 50, 1))
X = np.array(X).reshape(-1, img_size, img_size, 1)

#divide by 255 to squish values to 0 - 1
X = X/255.0
y = np.array(y)

#we made X and y np arrays to be able to feed into the model
test_pic = []
test_lab = []


for i in test_data:
    test_pic.append(i[0])
    test_lab.append(i[1])
        
#print(X[0].reshape(-1, 50, 50, 1))
test_pic = np.array(test_pic).reshape(-1, img_size, img_size, 1)


#divide by 255 to squish values to 0 - 1
test_pic = test_pic/255.0
test_lab = np.array(test_lab)

model = Sequential()

model.add(Conv2D(192,(5,3), input_shape=X.shape[1:], activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.6))

model.add(Conv2D(64,(3,3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.6))

model.add(Flatten()) 
model.add(Dense(192))

model.add(Activation("softmax")) 

opt = keras.optimizers.Adam(learning_rate=0.0002)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ['accuracy'])

history = model.fit(X, y, batch_size = 32, epochs = 70, verbose = 1, validation_split = 0.2)


model.save('my_cnn', save_format='tf')

model2 = keras.models.load_model('my_cnn') 
print(model2.summary())

yhat_probs = model2.predict(test_pic, verbose=0)
# predict crisp classes for test set
yhat_classes = model2.predict_classes(test_pic, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_lab, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_lab, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_lab, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_lab, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(test_lab, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(test_lab, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(test_lab, yhat_classes)
print(matrix)

# list all data in history
print(history.history.keys())
print(history.history)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

