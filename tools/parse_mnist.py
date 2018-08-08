import struct
import os
import numpy as np
import cv2

def run_0():
    mnist_im_path="../examples/mnist/t10k-images-idx3-ubyte"
    mnist_label_path="../examples/mnist/t10k-labels-idx1-ubyte"
    folder_save_dir = "../data/mnist/"

    f_labels = open(mnist_label_path,"rb")
    magic,num = struct.unpack(">2I",f_labels.read(8))
    labels = np.fromfile(f_labels,dtype=np.uint8)

    f_im = open(mnist_im_path,"rb")
    magic,num,rows,cols = struct.unpack(">4I",f_im.read(16))

    ims = np.fromfile(f_im,dtype=np.uint8).reshape([num,rows,cols])

    num = 0
    for label,im in zip(labels,ims):
        pic_save_dir = folder_save_dir+str(label)+"/"
        if not os.path.exists(pic_save_dir):
            os.makedirs(pic_save_dir)

        cv2.imwrite(pic_save_dir+"%06d.png"%num,im)
        num += 1

def run_1():
    pic_dir = "/home/zqp/picture/mnist/0/"
    for picname in os.listdir(pic_dir):
        im = cv2.imread(pic_dir+picname,-1)
        print(im.shape)

if __name__=="__main__":
    run_0()