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
        assert len(self._bottoms)==1,"%s bottom must equal 1: error %s"%(self._name,self._bottoms)
        assert self._bottoms[0] in blobs.keys(),"%s bottom %s is not init"%(self._name,self._bottoms[0])
        assert len(self._tops)==1,"%s top must equal 1: error %s"%(self._name,self._tops)
        assert self._tops[0] not in blobs.keys(),"%s top %s has been inited"%(self._name,self._tops[0])

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)>1,"error bottom %s"%bottom_shape

        self.__extract_parameter(parameter.inner_product_param)

        self._w = np.zeros([self.__num_output,reduce(lambda x,y: x*y,bottom_shape[1:])],dtype=np.float32)
        self._b = np.zeros([self.__num_output],dtype=np.float32)

        top_shape = self.__generate_top_shape(bottom_shape)
        blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,top_shape))

    def __extract_parameter(self,inner_product_param):
        self.__num_output = inner_product_param.num_output

    def __generate_top_shape(self,bottom_shape):
        out_n = bottom_shape[0]
        out_c = self.__num_output
        assert out_c>0 and out_n>0,"output feature map error:[%s,%s]"%(out_n,out_c)

        return (out_n,out_c)

    def forward(self,blobs):
        assert len(blobs[self._bottoms[0]].shape)>1,"%s error input blobs"%self._name
        bottom_blob = blobs[self._bottoms[0]].reshape((blobs[self._bottoms[0]].shape[0],-1))
        top_blob = blobs[self._tops[0]]

        assert bottom_blob.shape[1]==self._w.shape[1],"%s error input blobs"%self._name

        top_shape = self.__generate_top_shape(bottom_blob.shape)
        if top_blob.shape != top_shape:
            blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)
            sys.stderr.write("reshape blob %s from %s-->%s\n"%(self._tops[0],top_blob.shape,top_shape))
            top_blob = blobs[self._tops[0]]

        top_blob[...] = np.dot(bottom_blob,self._w.transpose()) + self._b

        sys.stderr.write("%s forward successfully\n"%self._name)

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = InnerProductLayer(parameter)

