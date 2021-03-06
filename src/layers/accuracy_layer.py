import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class AccuracyLayer(Layer):
    def __init__(self,parameter,blobs):
        super(AccuracyLayer, self).__init__(parameter)

    def forward(self,blobs):
        print("convolution forward")
        pass

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = AccuracyLayer(parameter)

