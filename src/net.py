from google.protobuf import text_format
import numpy as np
import sys
from layer import Layer
from proto import caffe_pb2
from layers.data_layer import DataLayer
from layers.conv_layer import ConvolutionLayer
class Net:
    def __init__(self,prototxt=None,parameter=None):
        if prototxt:
            parameter = caffe_pb2.NetParameter()
            text_format.Merge(open(prototxt).read(),parameter)

        if not parameter:
            sys.stderr.write("error init\n")

        self.__register_layers = ["Data","Convolution"]
        self.__parameter = parameter
        self.__blobs  = {}
        self.__layers = []
        self.__layer_names = []

        self.__init_from_parameter()
        print(self.__layer_names)

    def __init_from_parameter(self):
        if self.__parameter.input:
            self.__blobs[self.__parameter.input[0]] = None

        for layer in self.__parameter.layer:
            if layer.type in self.__register_layers:
                for bottom in layer.bottom:
                    if bottom not in self.__blobs.keys():
                        sys.stderr.write("unknown blob: %s\n"%bottom)
                        sys.exit(1)

                for top in layer.top:
                    if top not in self.__blobs.keys():
                        self.__blobs[top] = None
                    else:
                        sys.stderr.write("blobs %s has been exist\n"%top)
                        sys.exit(1)

                self.__layers.append(eval("%sLayer"%layer.type)(layer))
                self.__layer_names.append(layer.name)
            else:
                sys.stderr.write("unknown layer %s\n"%layer.type)
                sys.exit(1)

    def load_weights(self,weight_file):
        net_parameter = caffe_pb2.NetParameter()
        net_parameter.ParseFromString(open(weight_file,"rb").read())

    def input(self,input):
        assert isinstance(input,dict),"input is not dict"
        for key in input.keys():
            if key in self.__blobs.keys():
                self.__blobs[key] = input[key]
            else:
                sys.stderr.write("blob %s is not exist\n"%key)
                sys.exit(1)

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
    #text_format.Merge(open(prototxt).read(),net_parameter)


    #text_format.PrintMessage(net_parameter,open("test1.weights","w"))
