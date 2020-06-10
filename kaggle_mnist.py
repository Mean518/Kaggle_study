import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
print(train)
'''
x = []
y = []
for i in range(42000) :
    a = train.iloc[i,:]
    aa = a[1:]
    x.append([aa])
    bb = a[0]
    y.append(bb)
    
x = np.array(x)
y = np.array(y)

x = x.reshape(42000,28,28)

x_train = x[:40000]
x_test = x[-20000:]
y_train = y[:40000]
y_test = y[-20000:]

# 전처리 시작
x_train = (x_train/255).reshape(40000, 28, 28, 1)
x_test = (x_test/255).reshape(20000, 28, 28, 1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 모델구성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 성능 훨씬 좋음 -> 사실상 binary를 사용하는게 맞다는거임
hist = model.fit(x_train,y_train, epochs=145, batch_size=900, validation_split=0.1, verbose=2)  

# 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)
'''