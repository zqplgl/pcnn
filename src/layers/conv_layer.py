import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class ConvolutionLayer(Layer):
    def __init__(self,parameter):
        super(ConvolutionLayer, self).__init__(parameter)
        self.__convolution_param = parameter.convolution_param

    def forward(self,blobs):
        print("convolution forward")
        pass

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = ConvolutionLayer(parameter)


