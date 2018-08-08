import numpy as np
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class SoftmaxLayer(Layer):
    def __init__(self,parameter,blobs):
        super(SoftmaxLayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"bottom not equal 1"
        assert self._bottoms[0] in blobs.keys(),"bottom %s is not init"%self._bottoms[0]
        assert len(self._tops)==1,"top not equal 1"

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)>1,"error bottom %s"%bottom_shape

        if self._tops[0] not in blobs.keys():
            blobs[self._tops[0]] = np.zeros(bottom_shape,dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def forward(self,blobs):
        print("convolution forward")
        pass

if __name__=="__main__":
    #prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    #net = caffe_pb2.NetParameter()
    #text_format.Merge(open(prototxt).read(),net)

    #parameter = [layer for layer in net.layer if layer.type=="Convolution"][0]

    #conv = ConvolutionLayer(parameter)

    a = [1,2,3,4]
    b = np.array(a,dtype=np.float32)
    print(b)

