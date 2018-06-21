import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from sklearn.utils import shuffle

data=pd.read_csv('/home/my/Desktop/resume/percep/dog_class/labels.csv')
ds=data.values
data.tail(n=3)

path='train'

##mapping the names of breeds to images of dogs------------------------------
train=[]
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.jpg'): 
            name=filename[0:-4]
            for ix in range(ds.shape[0]):
                if ds[ix][0]==name:
                    train.append(ds[ix][1])


##saving the images as numpy arrays------------------------                    
train2 =[]
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            str1=path+'/'+filename
            im=Image.open(str1)
            im = im.resize((224,224))
            img=np.array(im)
            #img=img.reshape((-1, 225, 225, 3))
            #print img.shape
            train2.append(img)
            

t1=np.asarray(train)
t2=np.asarray(train2)

df = pd.DataFrame(t1)
df.to_csv("dogb.csv")

colnames=['extra','breed'] 
ds = pd.read_csv('dogb.csv', names=colnames, header=None)
print type(ds)
#ds.drop('extra')
df = ds.drop('extra', axis=1)
df=df.drop([0])
print df.head(n=1)

###using labelencoder to convert dog_breeds to value between 0 and n_classes-1.-----------------------------
lb_make = LabelEncoder()
df['breed'] = lb_make.fit_transform(data['breed'])
print df.head(n=5)
        
Y = np_utils.to_categorical(df)
print np.unique(df)
print Y.shape
####train-test split of data--------------------------------------------
split = int(0.8*data.shape[0])
print t2[:split].shape
print t2[split:].shape
X_train = t2[:split]
X_test = t2[split:]

Y_train = Y[:split]
Y_test = Y[split:]

print Y_train.shape
print X_test.shape

num_classes=len(np.unique(df))

##using the Resnet50 model for transfer learning----------------------------------
image_input = Input(shape=(224,224,3))

model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
model.summary()
###fine tuning the model------------------------------------
last_layer = model.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(128,activation='relu',name='fc1')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)

custom_resnet = Model(image_input,out)

for layer in custom_resnet.layers[:-3]:
    layer.trainable = False
custom_resnet.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

custom_resnet.fit(X_train,Y_train,epochs=12,shuffle=True,batch_size=64,validation_data=(X_test,Y_test))
custom_resnet.summary()
#####3
#K.clear_session()

