from functools import reduce
import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class InnerProductLayer(Layer):
    def __init__(self,parameter,blobs):
        super(InnerProductLayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"bottom not equal 1"
        assert self._bottoms[0] in blobs.keys(),"bottom %s is not init"%self._bottoms[0]
        assert len(self._tops)==1,"top not equal 1"
        assert self._tops[0] not in blobs.keys(),"top %s has been inited"%self._tops[0]

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)>1,"error bottom %s"%bottom_shape

        self.__extract_parameter(parameter.inner_product_param)

        self._w = np.zeros([self.__num_output,reduce(lambda x,y: x*y,bottom_shape[1:])],dtype=np.float32)
        self._b = np.zeros([self.__num_output],dtype=np.float32)


        out_n = bottom_shape[0]
        out_c = self.__num_output
        assert out_c>0 and out_n>0,"output feature map error:[%s,%s]"%(out_n,out_c)
        blobs[self._tops[0]] = np.zeros([out_n,out_c],dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def __extract_parameter(self,inner_product_param):
        self.__num_output = inner_product_param.num_output


    def forward(self,blobs):
        print("%s forward"%self._name)
        pass

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = InnerProductLayer(parameter)

