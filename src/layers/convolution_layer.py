import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import layer
from proto import caffe_pb2


class convolution_layer(layer):
    def __init__(self,parameter):
        super(convolution_layer, self).__init__(parameter)
        self.__convolution_param = parameter.convolution_param
        print(self._name)
        print(self._type)
        print(self.__convolution_param)

    def forward(self):
        pass

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = convolution_layer(parameter)


