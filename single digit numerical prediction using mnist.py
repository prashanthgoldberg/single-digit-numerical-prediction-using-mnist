#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical as t
from keras.models import Sequential


# In[12]:


(trainx,trainy),(testx,testy)=mnist.load_data()
trainx= trainx.reshape((trainx.shape[0], 28, 28, 1))
testx = testx.reshape((testx.shape[0], 28, 28, 1))
import numpy as np
np.shape(trainx)


# In[21]:


a=Sequential()
a.add(Conv2D(32,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(28,28,1)))
a.add(MaxPooling2D(2,2))
a.add(Conv2D(64,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(Conv2D(128,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(MaxPooling2D(2,2))
a.add(Flatten())
a.add(Dense(128,activation="relu",kernel_initializer="he_uniform"))
a.add(Dense(10,activation="softmax"))
opt=SGD(lr=0.001,momentum=0.9)
a.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# In[17]:


trainy,testy=t(trainy),t(testy)


# In[23]:


x1,x2=trainx.astype("float32"),testx.astype("float32")
trainx,testx=x1/255,x2/255


# In[24]:


a.fit(trainx,trainy,epochs=10,batch_size=32)


# In[25]:


a.save("mnist.h5")


# In[46]:


import cv2
img=cv2.imread("img.jpg")


# In[47]:


img=(img.astype("float32"))/255


# In[48]:


np.shape(img)


# In[50]:


img=img.reshape((3,28,28,1))


# In[54]:


d=a.predict(testx)


# In[12]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

img = load_image('sample_image.png')
a=load_model("mnist.h5")
digit = a.predict_classes(img)
print(digit[0])


# In[13]:


print(digit)


# In[6]:


print(a.predict(img))


# In[ ]:




