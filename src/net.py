from google.protobuf import text_format
import numpy as np
from layer import layer
from proto import caffe_pb2
class Net:
    def __init__(self,parameter):
        self.__parameter = parameter
        self.__blobs  = {}
        self.__layers = []

        if self.__parameter.input:
            self.__blobs[self.__parameter.input[0]] = np.array((self.__parameter.input_dim),dtype=np.float32)

        for layer in self.__parameter.layer:
            print(layer.type)


        print(self.__blobs.keys())



    def forward_layer(self,index,input=None):
        pass


if __name__=="__main__":
    prototxt = "/home/zqp/github/pcnn/examples/mnist/deploy.prototxt"
    net_parameter = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net_parameter)

    net = Net(net_parameter)



