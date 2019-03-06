#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#imgfileサイズ依存しないように改変
#original codeでは、予め（128,128.3）で格納しておく必要がある

import numpy as np
import glob
import h5py

from random import randrange

from keras.preprocessing.image import load_img, img_to_array

def main():
    img_rows =256
    img_cols =256

    finders = glob.glob("images/*")
    print(finders)
    imgs = []
    gimgs = []
    for finder in finders:
        files = glob.glob(finder+'/*')
        for imgfile in files:
            #img = load_img(imgfile, target_size=(img_rows,img_cols))
            #imgarray = img_to_array(img)
            #imgs.append(imgarray)
            
            grayimg = load_img(imgfile, grayscale=True, target_size=(img_rows,img_cols))
            grayimgarray = img_to_array(grayimg)
            #gimgs.append(grayimgarray)                 
            imgs.append(grayimgarray)
            
            
            #lackimg = imgarray.copy()
            lackimg = grayimgarray.copy()
            a = randrange(30)
            b = randrange(30)
            start_a = randrange(255-a-20)
            start_b = randrange(255-b-20)
            for i in range(a+20):
                for j in range(b+20):
                    lackimg[start_a + i][start_b + j][0] = 255
#                    for rgb in range(3):
#                        lackimg[start_a + i][start_b + j][rgb] = 255
            gimgs.append(lackimg)   
            

    # ただシャッフルしてtrainとvalに分けてるだけ
    perm = np.random.permutation(len(imgs))
    imgs = np.array(imgs)[perm]
    gimgs = np.array(gimgs)[perm]
    threshold = len(imgs)//10*9
    vimgs = imgs[threshold:]
    vgimgs = gimgs[threshold:]
    imgs = imgs[:threshold]
    gimgs = gimgs[:threshold]
    print('shapes')
    print('gen imgs : ', imgs.shape)
    print('raw imgs : ', gimgs.shape)
    print('val gen  : ', vimgs.shape)
    print('val raw  : ', vgimgs.shape)

    outh5 = h5py.File("datasetimages.hdf5", 'w')
    outh5.create_dataset('train_data_gen', data=imgs)
    outh5.create_dataset('train_data_raw', data=gimgs)
    outh5.create_dataset('val_data_gen', data=vimgs)
    outh5.create_dataset('val_data_raw', data=vgimgs)
    outh5.flush()
    outh5.close()


if __name__=='__main__':
    main()

