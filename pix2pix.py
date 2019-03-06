#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#tommyfms2/pix2pix-keras-byt より


import os
import argparse

import numpy as np

import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

from keras.utils import plot_model

import tensorflow as tf
from keras.utils import multi_gpu_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)


import models

def my_normalization(X):
    return X / 127.5 - 1
def my_inverse_normalization(X):
    return (X + 1.) / 2.

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = my_inverse_normalization(X_raw)
    X_proc = my_inverse_normalization(X_proc)
    X_gen = my_inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs,Xg,Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./figures/current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

# tmp load data gray to color
def my_load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_gen"][:].astype(np.float32)
        X_full_train = my_normalization(X_full_train)
        X_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        X_sketch_train = my_normalization(X_sketch_train)
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val


def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


def my_train(args):
    # load data
    procImage, rawImage, procImage_val, rawImage_val = my_load_data(datasetpath)
    print('procImage.shape : ', procImage.shape)
    print('rawImage.shape : ', rawImage.shape)
    print('procImage_val : ', procImage_val.shape)
    print('rawImage_val : ', rawImage_val.shape)

    img_shape = rawImage.shape[-3:]
    print('img_shape : ', img_shape)
    patch_num = (img_shape[0] // patch_size) * (img_shape[1] // patch_size)
    disc_img_shape = (patch_size, patch_size, procImage.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    #generator_model.load_weights('params_generator_starGAN_epoch_400.hdf5')
    # load discriminator model
    discriminator_model = models.my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    #discriminator_model.load_weights('params_discriminator_starGAN_epoch_400.hdf5')
    #generator_model.compile(loss='mae', optimizer=opt_discriminator)                                                                                                                                       
    generator_model.compile(loss='mean_squared_error', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.my_load_DCGAN(generator_model, discriminator_model, img_shape, args.patch_size)

    #loss = [l1_loss, 'binary_crossentropy']                                                                                                                                                                
    loss = ['mean_squared_error', 'binary_crossentropy']

    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
    plot_model(DCGAN_model, to_file = 'DCGAN.png')
    
    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
    plot_model(discriminator_model, to_file = 'discriminator.png')
    
    # start training
    j=0
    print('start training')
    for e in range(epoch):

        starttime = time.time()
        perm = np.random.permutation(rawImage.shape[0])
        X_procImage = procImage[perm]
        X_rawImage  = rawImage[perm]
        X_procImageIter = [X_procImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        X_rawImageIter  = [X_rawImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage.shape[0], batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for visualization　file名に通し番号を記載して残す
            if b_it % (procImage.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, batch_size, "training" +str(j))
                idx = np.random.choice(procImage_val.shape[0], batch_size)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation"+str(j))
                j += 1
        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, epoch, time.time() - starttime))
        generator_model.save_weights('params_generator_starGAN_epoch_{0:03d}.hdf5'.format(e+1), True)
        discriminator_model.save_weights('params_discriminator_starGAN_epoch_{0:03d}.hdf5'.format(e+1), True)


if __name__=='__main__':
    args = argparse.ArgumentParser(description='Train Font GAN')
    datasetpath = "datasetimages.hdf5"
    patch_size = 64
    batch_size = 32
    epoch = 1

    #args = parser
        

    K.set_image_data_format("channels_last")

    my_train(args)
