# EIP_A3

## Final Validation score of Base Model
Accuracy on test data is: 82.19

## Model Definition

model = Sequential()

model.add(SeparableConvolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3))) # Output dim : 32 x 32 x 32 ; Receptive field : 3 x 3

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConvolution2D(64, 3, 3)) # Output dim : 30 x 30 x 64 ; Receptive Field : 5 x 5

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # Output dim : 15 x 15 x 64 ; Receptive Field : 6 x 6

model.add(Dropout(0.1))

model.add(SeparableConvolution2D(128, 3, 3, border_mode='same')) # Output dim : 15 x 15 x 128 ; Receptive Field : 10 x 10

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConvolution2D(128, 3, 3)) # Output dim : 13 x 13 x 128 ; Receptive Field : 14 x 14

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # Output dim : 6 x 6 x 128 ; Receptive Field : 16 x 16

model.add(Dropout(0.1))

model.add(SeparableConvolution2D(192, 3, 3, border_mode='same')) # Output dim : 6 x 6 x 192 ; Receptive Field : 24 x 24

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConvolution2D(192, 3, 3)) # Output dim : 4 x 4 x 192 ; Receptive Field : 32 x 32

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(AveragePooling2D(pool_size=(4, 4))) # Output dim : 1 x 1 x 192 ; Receptive Field : 44 x 44

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))

## 50 Epochs of Model 

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
  1/390 [..............................] - ETA: 1:08 - loss: 0.8102 - acc: 0.7109/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:17: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:17: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, callbacks=[<keras.ca..., steps_per_epoch=390, epochs=50)`
390/390 [==============================] - 43s 111ms/step - loss: 0.7792 - acc: 0.7287 - val_loss: 0.8093 - val_acc: 0.7381
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 43s 110ms/step - loss: 0.7377 - acc: 0.7420 - val_loss: 0.7455 - val_acc: 0.7555
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 43s 111ms/step - loss: 0.7002 - acc: 0.7560 - val_loss: 0.5949 - val_acc: 0.8012
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 43s 111ms/step - loss: 0.6787 - acc: 0.7648 - val_loss: 0.6122 - val_acc: 0.7937
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 43s 110ms/step - loss: 0.6625 - acc: 0.7682 - val_loss: 0.5958 - val_acc: 0.8030
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 43s 110ms/step - loss: 0.6376 - acc: 0.7778 - val_loss: 0.6439 - val_acc: 0.7959
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 43s 110ms/step - loss: 0.6256 - acc: 0.7800 - val_loss: 0.6149 - val_acc: 0.7950
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 43s 110ms/step - loss: 0.6149 - acc: 0.7878 - val_loss: 0.5702 - val_acc: 0.8112
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 43s 110ms/step - loss: 0.6031 - acc: 0.7896 - val_loss: 0.5446 - val_acc: 0.8193
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 42s 109ms/step - loss: 0.6028 - acc: 0.7893 - val_loss: 0.5443 - val_acc: 0.8207
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 42s 109ms/step - loss: 0.5870 - acc: 0.7972 - val_loss: 0.5600 - val_acc: 0.8183
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 43s 110ms/step - loss: 0.5830 - acc: 0.7970 - val_loss: 0.5065 - val_acc: 0.8337
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 43s 110ms/step - loss: 0.5730 - acc: 0.8010 - val_loss: 0.5761 - val_acc: 0.8112
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 43s 110ms/step - loss: 0.5708 - acc: 0.8018 - val_loss: 0.5797 - val_acc: 0.8154
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 43s 110ms/step - loss: 0.5592 - acc: 0.8045 - val_loss: 0.5115 - val_acc: 0.8310
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 43s 109ms/step - loss: 0.5615 - acc: 0.8039 - val_loss: 0.5357 - val_acc: 0.8271
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 43s 109ms/step - loss: 0.5600 - acc: 0.8058 - val_loss: 0.5311 - val_acc: 0.8258
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 43s 110ms/step - loss: 0.5511 - acc: 0.8063 - val_loss: 0.5295 - val_acc: 0.8276
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 43s 109ms/step - loss: 0.5502 - acc: 0.8098 - val_loss: 0.5124 - val_acc: 0.8294
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 43s 109ms/step - loss: 0.5450 - acc: 0.8103 - val_loss: 0.5302 - val_acc: 0.8278
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 42s 109ms/step - loss: 0.5417 - acc: 0.8099 - val_loss: 0.5189 - val_acc: 0.8297
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 42s 109ms/step - loss: 0.5412 - acc: 0.8124 - val_loss: 0.5151 - val_acc: 0.8326
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 42s 109ms/step - loss: 0.5382 - acc: 0.8120 - val_loss: 0.5229 - val_acc: 0.8288
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 42s 109ms/step - loss: 0.5276 - acc: 0.8165 - val_loss: 0.5022 - val_acc: 0.8324
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 42s 108ms/step - loss: 0.5292 - acc: 0.8147 - val_loss: 0.5131 - val_acc: 0.8329
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 42s 109ms/step - loss: 0.5289 - acc: 0.8164 - val_loss: 0.4875 - val_acc: 0.8379
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 42s 109ms/step - loss: 0.5243 - acc: 0.8166 - val_loss: 0.5060 - val_acc: 0.8351
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 42s 108ms/step - loss: 0.5241 - acc: 0.8183 - val_loss: 0.4879 - val_acc: 0.8413
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 42s 109ms/step - loss: 0.5221 - acc: 0.8164 - val_loss: 0.4870 - val_acc: 0.8420
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 42s 108ms/step - loss: 0.5221 - acc: 0.8178 - val_loss: 0.5276 - val_acc: 0.8268
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 42s 108ms/step - loss: 0.5182 - acc: 0.8180 - val_loss: 0.4989 - val_acc: 0.8362
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 42s 109ms/step - loss: 0.5092 - acc: 0.8240 - val_loss: 0.4794 - val_acc: 0.8424
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 42s 109ms/step - loss: 0.5123 - acc: 0.8197 - val_loss: 0.5102 - val_acc: 0.8336
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 42s 108ms/step - loss: 0.5146 - acc: 0.8194 - val_loss: 0.4917 - val_acc: 0.8392
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 42s 108ms/step - loss: 0.5083 - acc: 0.8225 - val_loss: 0.4909 - val_acc: 0.8411
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 42s 109ms/step - loss: 0.5073 - acc: 0.8221 - val_loss: 0.4851 - val_acc: 0.8391
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 42s 109ms/step - loss: 0.5039 - acc: 0.8230 - val_loss: 0.5029 - val_acc: 0.8350
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 42s 109ms/step - loss: 0.5021 - acc: 0.8247 - val_loss: 0.5072 - val_acc: 0.8370
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 42s 109ms/step - loss: 0.5051 - acc: 0.8218 - val_loss: 0.4833 - val_acc: 0.8425
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 43s 109ms/step - loss: 0.5040 - acc: 0.8258 - val_loss: 0.4959 - val_acc: 0.8404
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 42s 109ms/step - loss: 0.5089 - acc: 0.8216 - val_loss: 0.4841 - val_acc: 0.8442
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 43s 109ms/step - loss: 0.4989 - acc: 0.8246 - val_loss: 0.4962 - val_acc: 0.8404
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 43s 109ms/step - loss: 0.4999 - acc: 0.8254 - val_loss: 0.4857 - val_acc: 0.8422
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 43s 109ms/step - loss: 0.5022 - acc: 0.8252 - val_loss: 0.4846 - val_acc: 0.8463
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 42s 109ms/step - loss: 0.5001 - acc: 0.8255 - val_loss: 0.4962 - val_acc: 0.8412
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 42s 109ms/step - loss: 0.4960 - acc: 0.8257 - val_loss: 0.4787 - val_acc: 0.8463
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 42s 107ms/step - loss: 0.4970 - acc: 0.8279 - val_loss: 0.4878 - val_acc: 0.8411
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 42s 108ms/step - loss: 0.4970 - acc: 0.8271 - val_loss: 0.5021 - val_acc: 0.8380
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 42s 108ms/step - loss: 0.4995 - acc: 0.8240 - val_loss: 0.4959 - val_acc: 0.8384
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 42s 108ms/step - loss: 0.4935 - acc: 0.8282 - val_loss: 0.4878 - val_acc: 0.8424
Model took 2126.97 seconds to train

Accuracy on test data is: 84.24

## Team Members : Dinesh Kesaboina, Abhiraj Gupta Vinnakota
