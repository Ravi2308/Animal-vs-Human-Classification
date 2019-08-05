# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:40:26 2018

@author: lotus
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
train='C:/Users/lotus/Desktop/New folder (2)/train1'
test='C:/Users/lotus/Desktop/New folder (2)/test1'

imgsize=40
learning_rate=1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(learning_rate, '2conv-basic')

def label_img(img):
    word=img.split('.')[-3]
    if word=='cat':return[1,0]
    elif word=='dog':return[0,1]
    
def train_data():
    training_data=[]
    for img  in tqdm(os.listdir(train)):
        label=label_img(img)
        path=os.path.join(train,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(40,40))
        training_data.append([np.array(img),np.array(label)])
        
    shuffle(training_data)
    np.save('train.npy',training_data)
    return training_data

traindata=train_data()

def test_data():
    testing_data=[]
    for img in tqdm(os.listdir(test)):
        path=os.path.join(test,img)
        img_num=img.split('.')[0]
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(40,40))
        testing_data.append([np.array(img),img_num])
        
    np.save('test.npy',testing_data)
    return testing_data

testdata=test_data()
import tflearn

net=tflearn.input_data(shape=[None,1])
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,64)
net=tflearn.fully_connected(net,1024)
net=tflearn.fully_connected(net,2,activation='softmax')
net=tflearn.regression(net)

model=tflearn.DNN(net,tensorboard_dir='log')
#model.fit(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

#model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = traindata[:-500]
test = traindata[-500:]

X = np.array([i[0] for i in train]).reshape(-1,40,40,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,40,40,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(testdata[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(40,40,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

