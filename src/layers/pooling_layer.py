import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class PoolingLayer(Layer):
    def __init__(self,parameter,blobs):
        super(PoolingLayer, self).__init__(parameter)
        self.__pool_method = {0:"MAX"}
        assert len(self._bottoms)==1,"bottom not equal 1"
        assert self._bottoms[0] in blobs.keys(),"bottom %s is not init"%self._bottoms[0]
        assert len(self._tops)==1,"top not equal 1"
        assert self._tops[0] not in blobs.keys(),"top %s has been inited"%self._tops[0]

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)==4,"error bottom %s"%bottom_shape

        self.__extract_parameter(parameter.pooling_param)

        top_shape = self.__generate_top_shape(bottom_shape)
        blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def __generate_top_shape(self,bottom_shape):
        out_n = bottom_shape[0]
        out_c = bottom_shape[1]
        out_h = int((bottom_shape[2] + 2*self.__pad_h - self.__kernel_h)/self.__stride_h+1)
        out_w = int((bottom_shape[3] + 2*self.__pad_w - self.__kernel_w)/self.__stride_w+1)

        assert out_n>0 and out_c>0 and out_h>0 and out_w>0,"%s output feature map error:[%s,%s,%s,%s]"%(self._name,out_n,out_c,out_h,out_w)

        return (out_n,out_c,out_h,out_w)

    def __extract_parameter(self,pooling_param):
        self.__pool = pooling_param.pool

        assert self.__pool in self.__pool_method.keys(),"unknown pooling method in %s layer"%self._name

        self.__kernel_h = pooling_param.kernel_size
        self.__kernel_w = pooling_param.kernel_size

        self.__pad_h = pooling_param.pad
        self.__pad_w = pooling_param.pad

        self.__stride_w = pooling_param.stride
        self.__stride_h = pooling_param.stride

    def __max_pooling(self,bottom_blob,top_blob):
        top_n,top_c,top_h,top_w = top_blob.shape
        bottom_n,bottom_c,bottom_h,bottom_w = bottom_blob.shape

        for t_n in range(top_n):
            for t_c in range(top_c):
                for t_h in range(top_h):
                    for t_w in range(top_w):
                        for row in range(self.__kernel_h):
                            offset_h = t_h*self.__stride_h-self.__pad_h+row
                            if offset_h<0 or offset_h>=bottom_h:
                                continue
                            for col in range(self.__kernel_w):
                                offset_w = t_w*self.__stride_w-self.__pad_w+col
                                if offset_w<0 or offset_w>=bottom_w:
                                    continue

                                if row==0 and col==0:
                                    top_blob[t_n][t_c][t_h][t_w] = bottom_blob[t_n][t_c][offset_h][offset_w]
                                else:
                                    top_blob[t_n][t_c][t_h][t_w] = max(top_blob[t_n][t_c][t_h][t_w],bottom_blob[t_n][t_c][offset_h][offset_w])

    def forward(self,blobs):
        bottom_blob = blobs[self._bottoms[0]]
        top_blob = blobs[self._tops[0]]

        assert len(bottom_blob.shape)==4,"%s error input blobs"%self._name
        top_shape = self.__generate_top_shape(bottom_blob.shape)

        if top_blob.shape != top_shape:
            blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)
            sys.stderr.write("reshape blob %s from %s-->%s\n"%(self._tops[0],top_blob.shape,top_shape))
            top_blob = blobs[self._tops[0]]

        if self.__pool_method[self.__pool]=="MAX":
            self.__max_pooling(bottom_blob,top_blob)

        sys.stderr.write("%s forward successfully\n"%self._name)

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = PoolingLayer(parameter)

