import numpy as np
import platform
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import regularizers


"""
Deep Learning Custom Factors

*** normalize or scaler
*** loss & metrics

- number of hidden layers
- number of nodes in each layer
- activation function
- optimizer
- learning rate of optimizer
- epochs : earlyStopping 활용
- batch_size
- dropout

*** regularization :
- kernel_regularizer=regularizers.l1(0.01) / l2 / elastic_net

*** initializer
- random_uniform : -0.05 ~ 0.05 사이의 값으로 균등하게 초기화
- random_normal : mean 0, std 0.5인 가우시안 분포에 따라 초기화

"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_size = 28*28
n_classes = 10

X_train = X_train.reshape(60000, n_size).astype('float32')
X_test = X_test.reshape(10000, n_size).astype('float32')

X_train /= 255  # normalize
X_test  /= 255

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)


optimizer = Adam()
# optimizer = sgd()
# optimizer = RMSprop()

n_epoch = 300
batch_size = 256
n_hidden = 256
verbose = 1
validation_split = 0.1
dropout = 0.1

early_stopping_monitor = EarlyStopping(patience=10)

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_size, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(n_hidden, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(n_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=n_epoch,
                    callbacks=[early_stopping_monitor],
                    verbose=verbose, validation_split=validation_split)

score = model.evaluate(X_test, Y_test)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

print('')
print(model.predict_classes(X_test[:10]))
print('')
print(model.predict_proba(X_test[:10]))
print('')


"""
*** plot accuracy and loss
"""
if platform.system() == 'Windows':
    import matplotlib.pyplot as plt

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    print('acc', history.history['acc'])
    print('val_acc',history.history['val_acc'])


"""
*** save and restore trained model
"""
from keras.models import load_model, model_from_json

# 학습된 모델 저장
model.save('mnist_model.h5')

# 저장된 모델 불러오기
model = load_model('mnist_model.h5')


# 모델 구조만 저장
json_str = model.to_json()

# json 파일에 저장된 구조로 모델 재구성
model = model_from_json(json_str)
