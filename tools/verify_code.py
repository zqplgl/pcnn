import caffe
import numpy as np
import cv2

def print_blob(blob,text):
    assert len(blob.shape)==4
    f = open(text,"w")

    for n in range(blob.shape[0]):
        for c in range(blob.shape[1]):
            for h in range(blob.shape[2]):
                for w in range(blob.shape[3]):
                    f.write(str(blob[n][c][h][w])+"\n")

def process_im():
    im_path = "../data/mnist/0/000003.png"
    im = cv2.imread(im_path,-1)
    im = im.astype(np.float32)
    im = im.reshape([1,1,28,28])

    return im

def run():
    prototxt = "../examples/mnist/deploy.prototxt"
    weight_file = "../examples/mnist/weight.caffemodel"

    caffe.set_mode_gpu()
    net = caffe.Net(prototxt,weight_file,caffe.TEST)
    im = process_im()

    net.blobs["data"].reshape(*im.shape)
    net.blobs["data"].data[...] = im
    net.forward()

    #print(net.blobs["pool2"].data)
    for key in net.params.keys():
        print(key)

    print(net.blobs["ip2"].data.shape)
    print(net.blobs["ip2"].data)
    print(net.blobs["prob"].data)


if __name__=="__main__":
    run()