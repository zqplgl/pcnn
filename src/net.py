from google.protobuf import text_format
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

if __name__=="__main__":
    input = {}
    input["data"] = np.array([1,2,3],dtype=np.float32)
    prototxt = "/home/zqp/install_lib/models/mnist/deploy.prototxt"
    prototxt = "/home/zqp/github/pcnn/examples/mnist/deploy.prototxt"
    weight_file = "/home/zqp/install_lib/models/mnist/weight.caffemodel"

    net = Net(prototxt=prototxt)
    net.load_weights_from_caffemodel(weight_file)

    im = np.array([[[[1,2,3,4,5,6],[4,5,6,7,8,9],[1,2,3,4,5,6],[4,5,6,7,8,9],[1,2,3,4,5,6],[4,5,6,7,8,9]]]],dtype=np.float32)
    net.input({"data":im})
    net.forward_layer(0)

    for key in net.blobs:
        print(net.blobs[key])

    #text_format.PrintMessage(net_parameter,open("test1.weights","w"))

