from google.protobuf import text_format
import numpy as np
import sys
from layer import Layer
from proto import caffe_pb2
from layers.data_layer import DataLayer
from layers.conv_layer import ConvolutionLayer
class Net:
    def __init__(self,parameter):
        self.__register_layers = ["Data","Convolution"]
        self.__parameter = parameter
        self.__blobs  = {}
        self.__layers = []

        if self.__parameter.input:
            self.__blobs[self.__parameter.input[0]] = np.array((self.__parameter.input_dim),dtype=np.float32)

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
            else:
                sys.stderr.write("unknown layer %s\n"%layer.type)
                sys.exit(1)

        print("bolbs: ",self.__blobs.keys())

    def input(self,input):
        assert isinstance(input,dict),"input is not dict"
        for key in input.keys():
            if key in self.__blobs.keys():
                self.__blobs[key] = input[key]
            else:
                sys.stderr.write("blob %s is not exist\n"%key)
                sys.exit(1)


    def forward_layer(self,index=0):
        print(self.__blobs)
        for layer in self.__layers[index:]:
            layer.forward(self.__blobs)

if __name__=="__main__":
    input = {}
    input["data"] = np.array([1,2,3],dtype=np.float32)
    prototxt = "/home/zqp/github/pcnn/examples/mnist/deploy.prototxt"
    net_parameter = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net_parameter)

    net = Net(net_parameter)
    net.input(input)
    net.forward_layer(0)
