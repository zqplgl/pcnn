from google.protobuf import text_format
import cv2
from collections import OrderedDict
import numpy as np
import sys
from layer import Layer
from proto import caffe_pb2
from layers.accuracy_layer import AccuracyLayer
from layers.conv_layer import ConvolutionLayer
from layers.data_layer import DataLayer
from layers.inner_product_layer import InnerProductLayer
from layers.pooling_layer import PoolingLayer
from layers.relu_layer import ReLULayer
from layers.softmax_layer import SoftmaxLayer
class Net:
    def __init__(self,prototxt=None,parameter=None):
        if prototxt:
            parameter = caffe_pb2.NetParameter()
            text_format.Merge(open(prototxt).read(),parameter)

        if not parameter:
            sys.stderr.write("error init\n")

        self.__register_layers = ["Accuracy","Convolution","Data","InnerProduct","Pooling","ReLU","Softmax"]
        self.__parameter = parameter
        self.__blobs  = {}
        self.__layers = []
        self.__layer_names = []

        self.__init_from_parameter()

    def __init_from_parameter(self):
        if self.__parameter.input and self.__parameter.input_dim:
            self.__blobs[self.__parameter.input[0]] = np.zeros(self.__parameter.input_dim,dtype=np.float32)

        for layer in self.__parameter.layer:
            if layer.type in self.__register_layers:
                self.__layers.append(eval("%sLayer"%layer.type)(layer,self.__blobs))
                self.__layer_names.append(layer.name)
            else:
                sys.stderr.write("unknown layer %s\n"%layer.type)
                sys.exit(1)

    def load_weights_from_caffemodel(self,weight_file):
        net_parameter = caffe_pb2.NetParameter()
        net_parameter.ParseFromString(open(weight_file,"rb").read())

        for layer in net_parameter.layer:
            if layer.name not in self.__layer_names:
                sys.stderr.write("ignore layer %s\n"%layer.name)
                continue

            l = self.__layers[self.__layer_names.index(layer.name)]

            if len(layer.blobs)==2:
                sys.stderr.write("loading parameter from %s\n"%layer.name)
                l.load_parameter(layer.blobs)

    @property
    def blobs(self):
        return self.__blobs

    def input(self,input):
        assert isinstance(input,dict),"input is not dict"
        for key in input.keys():
            if key in self.__blobs.keys():
                self.__blobs[key] = input[key]
            else:
                sys.stderr.write("blob %s is not exist\n"%key)
                sys.exit(1)

    def reshape(self,blob_shape):
        pass

    def forward_layer(self,index=0):
        for layer in self.__layers[index:]:
            layer.forward(self.__blobs)

def process_im():
    im_path = "../data/mnist/0/000003.png"
    im = cv2.imread(im_path,-1)
    im = im.astype(np.float32)
    im = im.reshape([1,1,28,28])

    return im

def print_blob(blob,text):
    assert len(blob.shape)==4
    f = open(text,"w")

    for n in range(blob.shape[0]):
        for c in range(blob.shape[1]):
            for h in range(blob.shape[2]):
                for w in range(blob.shape[3]):
                    f.write(str(blob[n][c][h][w])+"\n")

def load_from_txt():
    f = open("../tools/verify_pool1.txt")

    pool1 = [float(num) for num in f.read().split()]
    pool1 = np.array(pool1,dtype=np.float32).reshape([1,20,12,12])

    print(pool1.shape)
    return pool1


if __name__=="__main__":
    input = {}
    input["data"] = np.array([1,2,3],dtype=np.float32)
    prototxt = "../examples/mnist/deploy.prototxt"
    weight_file = "../examples/mnist/weight.caffemodel"

    net = Net(prototxt=prototxt)
    net.load_weights_from_caffemodel(weight_file)

    im = process_im()
    net.input({"data":im})
    net.forward_layer()

    print(net.blobs["ip1"])


    #text_format.PrintMessage(net_parameter,open("test1.weights","w"))

