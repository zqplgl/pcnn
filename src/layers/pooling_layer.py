import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class PoolingLayer(Layer):
    def __init__(self,parameter,blobs):
        super(PoolingLayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"bottom not equal 1"
        assert self._bottoms[0] in blobs.keys(),"bottom %s is not init"%self._bottoms[0]
        assert len(self._tops)==1,"top not equal 1"
        assert self._tops[0] not in blobs.keys(),"top %s has been inited"%self._tops[0]

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)==4,"error bottom %s"%bottom_shape

        self.__extract_parameter(parameter.pooling_param)

        out_n = bottom_shape[0]
        out_c = bottom_shape[1]
        out_h = int((bottom_shape[2]+2*self.__pad_h-self.__kernel_h)/self.__stride_h+1)
        out_w = int((bottom_shape[3]+2*self.__pad_w-self.__kernel_w)/self.__stride_w+1)
        assert out_n>0 and out_c>0 and out_h>0 and out_w>0,"output feature map error:[%s,%s,%s,%s]"%(out_n,out_c,out_h,out_w)
        blobs[self._tops[0]] = np.zeros([out_n,out_c,out_h,out_w],dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def __extract_parameter(self,pooling_param):
        self.__pool = pooling_param.pool

        self.__kernel_h = pooling_param.kernel_size
        self.__kernel_w = pooling_param.kernel_size

        self.__pad_h = pooling_param.pad
        self.__pad_w = pooling_param.pad

        self.__stride_w = pooling_param.stride
        self.__stride_h = pooling_param.stride


    def forward(self,blobs):
        print("pooling forward")
        pass

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = PoolingLayer(parameter)

