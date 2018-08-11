import numpy as np
from google.protobuf import text_format
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class ReLULayer(Layer):
    def __init__(self,parameter,blobs):
        super(ReLULayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"%s bottom must equal 1: error %s"%(self._name,self._bottoms)
        assert self._bottoms[0] in blobs.keys(),"%s bottom %s is not init"%(self._name,self._bottoms[0])
        assert len(self._tops)==1,"%s top must equal 1: error %s"%(self._name,self._tops)

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)>1,"%s error bottom %s"%(self._name,bottom_shape)

        if self._tops[0] not in blobs.keys():
            blobs[self._tops[0]] = np.zeros(bottom_shape,dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def forward(self,blobs):
        bottom_blob = blobs[self._bottoms[0]]
        top_blob = blobs[self._tops[0]]

        if top_blob.shape!=bottom_blob.shape:
            blobs[self._tops[0]] = np.zeros(bottom_blob.shape,dtype=np.float32)
            top_blob = blobs[self._tops[0]]

        top_blob[...] = np.where(top_blob<0,0,top_blob)
        sys.stderr.write("%s forward successfully\n"%self._name)

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)

    parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    conv = ReLULayer(parameter)

