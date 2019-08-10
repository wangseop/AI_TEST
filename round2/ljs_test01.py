from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split


import numpy as np
import os
import tensorflow as tf
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)



# # seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# └ Data 양이 많고 전처리가 잘 되어있는 데이터 => 정확도가 높다
# └ 만일 data양을 줄인다면? 정확도가 떨어질 것이다

X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.994)
X_test, _, Y_test, _ = train_test_split(X_test, Y_test, test_size=0.97)


# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
X_train = X_train.reshape(X_train.shape[0], 32 ,32, 3)  
X_test = X_test.reshape(X_test.shape[0], 32 ,32, 3)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)      # 분류값 집어넣는다 (0 ~ 9), onehot encoding 방식 사용
Y_test = np_utils.to_categorical(Y_test)        # 0000001000 : 6 값을 의미
# print(Y_train.shape)
# print(Y_test.shape)


def build_network(keep_prob=0.1, optimizer='adam'):
    # 컨볼루션 신경망의 설정
    inputs = Input(shape=(32,32,3), name='input')
    x = Conv2D(30, kernel_size=(3,3), activation='relu')(inputs)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(40, (3,3), activation='relu')(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    # inputs = Input(shape=(32,32,3), name='input')
    # x = Conv2D(5, kernel_size=(3,3), activation='relu')(inputs)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = Conv2D(5, (4,4), activation='relu')(x)
    # x = MaxPooling2D(pool_size=4)(x)
    # x = Conv2D(5, (3,3), activation='relu')(x)
    # x = Dropout(keep_prob)(x)
    # x = Flatten()(x)
    # x = Dense(10, activation='relu')(x)
    # x = Dropout(keep_prob)(x)
    # prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.summary()

    model.compile(loss='categorical_crossentropy',  # 분류 모델은 loss로 categorical_crossentropy 사용
                optimizer=optimizer,
                metrics=['accuracy'])
    model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=256),
                        steps_per_epoch=len(X_train),     # 증폭시킬 양
                        epochs=1,
                        verbose=1)   #, callbacks=callbacks
    
    return model


def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.05,0.5, 10)
    epochs = [10,50,100,150, 200]
    return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

def data_scaling(x_data):
    x_data = x_data.reshape(x_data.shape[0], 32* 32* 3) 
    MinMaxScaler()
    x_data = x_data.reshape(x_data.shape[0], 32, 32, 3)  


# early_stopping_callback = EarlyStopping(monitor='loss', patience=20)

model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()


pipe = Pipeline([('minmax', data_scaling(X_train)), ('model', model)])
# pipe = Pipeline([('minmax', MinMaxScaler()), ('model', model)])
search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters,
                            n_iter=5, n_jobs=-1, cv=5)



search.fit(X_train, Y_train)

print("score :", search.best_params_)
print("score :", search.best_estimator_)
print("score :", search.score(X_test, Y_test))

