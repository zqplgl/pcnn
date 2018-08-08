import caffe
import cv2

def run():
    prototxt = "../examples/mnist/deploy.prototxt"
    weight_file = "../examples/mnist/weight.caffemodel"

    caffe.set_mode_gpu()
    net = caffe.Net(prototxt,weight_file,caffe.TEST)

    for key in net.blobs.keys():
        print(key)

    im_path = "../data/mnist/0/000003.png"
    im = cv2.imread(im_path,-1)


    print(im.shape)


if __name__=="__main__":
    run()