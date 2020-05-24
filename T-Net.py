from keras.layers import BatchNormalization, Activation, Conv2D, concatenate, MaxPooling2D, UpSampling2D, add, Input
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from skimage.io import imread
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# data processing
row = col = 256
filepath = 'D:/Desktop'
train_image_path = os.path.join(filepath, 'Aug-image')
images = os.listdir(train_image_path)
total = len(images)
imgs = np.ndarray((total, row, col), dtype=np.uint8)
i = 0
for image_name in images:
    img_mask = imread(os.path.join(train_image_path, image_name), as_gray=True)
    imgs[i] = np.array(img_mask)
    i += 1
imgs = imgs.reshape(-1, row, col, 1)
imgs = imgs.astype('float32')
imgs /= 255.

train_label_path = os.path.join(filepath, 'Aug-label')
label = os.listdir(train_label_path)
labels = np.ndarray((total, row, col), dtype=np.uint8)
i = 0
for label_name in label:
    img_mask = imread(os.path.join(train_label_path, label_name), as_gray=True)
    labels[i] = np.array(img_mask)
    i += 1
labels = labels.reshape(-1, row, col, 1)
labels = labels.astype('float32')
labels /= 255.

X_train, y_test, X_label, y_label = train_test_split(imgs, labels, test_size=0.1, random_state=1)

# define short_cut model
def short_cut(inpt, filter):
    x = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(inpt)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = add([x, inpt])
    x = Activation('relu')(x)
    return x

# define encoder
def encoder(inpt, filter):
    e1 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inpt)
    e2 = MaxPooling2D(pool_size=(2, 2))(e1)
    e3 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(e2)
    e3_2 = short_cut(e3, filter)
    e4 = MaxPooling2D(pool_size=(2, 2))(e3)
    e5 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(e4)
    e5_2 = short_cut(e5, filter)
    e6 = UpSampling2D(size=(2, 2))(e5)
    e7 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(e6)
    e7 = add([e3_2, e7])
    e7_2 = short_cut(e7, filter)
    e8 = MaxPooling2D(pool_size=(2, 2))(e7)
    e9 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(e8)
    e9 = add([e9, e5_2])
    e10 = UpSampling2D(size=(2, 2))(e9)
    y = add([e3_2, e7_2, e10])
    return e1, e3, e5, y

# define decoder
def decoder(inpt, filter, e1, e2, e3, e4, e5, e6):
    d1 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inpt)
    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = concatenate([d2, e1, e2, e3], axis=3)
    d3 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(d2)
    d3_2 = short_cut(d3, filter)
    d4 = UpSampling2D(size=(2, 2))(d3)
    d4 = concatenate([d4, e4, e5, e6], axis=3)
    d5 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(d4)
    d5_2 = short_cut(d5, filter)
    d6 = MaxPooling2D(pool_size=(2, 2))(d5)
    d7 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(d6)
    d7 = add([d3_2, d7])
    d7_2 = short_cut(d7, filter)
    d8 = UpSampling2D(size=(2, 2))(d7)
    d9 = Conv2D(filter, 3, padding='same', activation='relu', kernel_initializer='he_normal')(d8)
    d9 = add([d9, d5_2])
    d10 = MaxPooling2D(pool_size=(2, 2))(d9)
    y = add([d3_2, d7_2, d10])
    return d1, d3, d5, y

# define Loss
def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_pred_f*y_true_f)
    return -(2.*intersection+1)/(K.sum(y_true_f)+K.sum(y_pred_f))

def DSCLoss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)+0.1*categorical_crossentropy(y_true, y_pred)


x = Input(shape=(256, 256, 1))
# --------------encoder-------------------------
# E1
E1_1, E1_3, E1_5, E1_y = encoder(x, 32)
# E2
E2_1, E2_3, E2_5, E2_y = encoder(E1_y, 64)
# E3
E3_1, E3_3, E3_5, E3_y = encoder(E2_y, 128)
# E4
E4_1, E4_3, E4_5, E4_y = encoder(E3_y, 256)
# 5
E5_out = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E4_y)
E5 = MaxPooling2D(pool_size=(2, 2))(E5_out)
E5 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E5)
E5_1 = short_cut(E5, 512)
E5 = MaxPooling2D(pool_size=(2, 2))(E5)
E5 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E5)
E5_2 = short_cut(E5, 512)
E5 = UpSampling2D(size=(2, 2))(E5)
E5 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E5)
E5 = add([E5, E5_1])
E5 = MaxPooling2D(pool_size=(2, 2))(E5)
E5 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E5)
E5 = add([E5, E5_2])
E5 = UpSampling2D(size=(2, 2))(E5)

# center
cen = Conv2D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(E5)  # 8*8*1028

# ----------------------------decoder-----------------------
# D5
D5_1, D5_3, D5_5, D5_y = decoder(cen, 512, E5_out, E4_3, E3_5, E4_1, E3_3, E2_5)
# D4
D4_1, D4_3, D4_5, D4_y = decoder(D5_y, 256, E4_1, E3_3, E2_5, E1_5, E3_1, E2_3)
# D3
D3 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D4_y)
D3 = UpSampling2D(size=(2, 2))(D3)
D3 = concatenate([D3, E3_1, E2_3, E1_5])
D3 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D3)
D3_1 = short_cut(D3, 128)
D3 = UpSampling2D(size=(2, 2))(D3)
D3 = concatenate([D3, E2_1, E1_3])
D3 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D3)
D3_2 = short_cut(D3, 128)
D3 = MaxPooling2D(pool_size=(2, 2))(D3)
D3 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D3)
D3 = add([D3, D3_1])
D3_3 = short_cut(D3, 128)
D3 = UpSampling2D(size=(2, 2))(D3)
D3 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D3)
D3 = add([D3, D3_2])
D3 = MaxPooling2D(pool_size=(2, 2))(D3)
D3 = add([D3, D3_1, D3_3])
# D2
D2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D3)
D2 = UpSampling2D(size=(2, 2))(D2)
D2 = concatenate([D2, E2_1, E1_3])
D2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D2)
D2_1 = short_cut(D2, 64)
D2 = UpSampling2D(size=(2, 2))(D2)
D2 = concatenate([D2, E1_1])
D2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D2)
D2_2 = short_cut(D2, 64)
D2 = MaxPooling2D(pool_size=(2, 2))(D2)
D2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D2)
D2 = add([D2, D2_1])
D2_3 = short_cut(D2, 64)
D2 = UpSampling2D(size=(2, 2))(D2)
D2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D2)
D2 = add([D2, D2_2])
D2 = MaxPooling2D(pool_size=(2, 2))(D2)
D2 = add([D2, D2_1, D2_3])
# D1
D1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D2)
D1 = UpSampling2D(size=(2, 2))(D1)
D1 = concatenate([D1, E1_1])
D1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D1)
D1_1 = short_cut(D1, 32)
D1 = UpSampling2D(size=(2, 2))(D1)
D1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D1)
D1_2 = short_cut(D1, 32)
D1 = MaxPooling2D(pool_size=(2, 2))(D1)
D1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D1)
D1 = add([D1, D1_1])
D1_3 = short_cut(D1, 32)
D1 = UpSampling2D(size=(2, 2))(D1)
D1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D1)
D1 = add([D1, D1_2])
D1 = MaxPooling2D(pool_size=(2, 2))(D1)
D1 = add([D1, D1_1, D1_3])

opt = Conv2D(4, 3, padding='same', activation='relu', kernel_initializer='he_normal')(D1)
opt = Conv2D(1, 1, padding='same', activation='sigmoid', kernel_initializer='he_normal')(opt)

model = Model(x, opt)
model.compile(optimizer=Adam(lr=0.0001), loss=DSCLoss, metrics=['accuracy'])
result = model.fit(X_train, X_label, batch_size=16, epochs=100, verbose=1, validation_split=0.2, validation_data=(y_test, y_label))
plt.plot(np.arange(len(result.history['acc'])), result.history['acc'], label='training')
plt.plot(np.arange(len(result.history['val_acc'])), result.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

predict = model.predict(y_test)
plt.subplot(331)
plt.imshow(((predict[:, :, 0])*255.).astype(np.uint8))
plt.subplot(332)
plt.imshow(((y_label[:, :, 0])*255.).astype(np.uint8))
plt.subplot(333)
plt.imshow(((y_test[:, :, 0])*255.).astype(np.uint8))
plt.subplot(334)
plt.imshow(((predict[:, :, 1])*255.).astype(np.uint8))
plt.subplot(335)
plt.imshow(((y_label[:, :, 1])*255.).astype(np.uint8))
plt.subplot(336)
plt.imshow(((y_test[:, :, 1])*255.).astype(np.uint8))
plt.subplot(337)
plt.imshow(((predict[:, :, 2])*255.).astype(np.uint8))
plt.subplot(338)
plt.imshow(((y_label[:, :, 2])*255.).astype(np.uint8))
plt.subplot(334)
plt.imshow(((y_test[:, :, 2])*255.).astype(np.uint8))

plt.show()




