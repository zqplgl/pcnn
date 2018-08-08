import numpy as np
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2


class ConvolutionLayer(Layer):
    def __init__(self,parameter,blobs):
        super(ConvolutionLayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"bottom not equal 1"
        assert self._bottoms[0] in blobs.keys(),"bottom %s is not init"%self._bottoms[0]
        assert len(self._tops)==1,"top not equal 1"
        assert self._tops[0] not in blobs.keys(),"top %s has been inited"%self._tops[0]

        self.__extract_parameter(parameter.convolution_param)

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)==4,"error bottom %s"%bottom_shape

        kernel_c = bottom_shape[1]
        self._w = np.zeros([self.__num_output,kernel_c,self.__kernel_h,self.__kernel_w],dtype=np.float32)
        self._b = np.zeros([self.__num_output],dtype=np.float32)

        out_n = bottom_shape[0]
        out_c = self.__num_output
        out_h = int((bottom_shape[2] + 2*self.__pad_h - self.__kernel_h)/self.__stride_h+1)
        out_w = int((bottom_shape[3] + 2*self.__pad_w - self.__kernel_w)/self.__stride_w+1)

        assert out_n>0 and out_c>0 and out_h>0 and out_w>0,"output feature map error:[%s,%s,%s,%s]"%(out_n,out_c,out_h,out_w)
        blobs[self._tops[0]] = np.zeros([out_n,out_c,out_h,out_w],dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def __extract_parameter(self,convolution_param):
        self.__num_output = convolution_param.num_output

        if len(convolution_param.kernel_size):
            self.__kernel_h = convolution_param.kernel_size[0]
            self.__kernel_w = convolution_param.kernel_size[0]

        if len(convolution_param.pad):
            self.__pad_h = convolution_param.pad[0]
            self.__pad_w = convolution_param.pad[0]

        else:
            self.__pad_h = convolution_param.pad_h
            self.__pad_w = convolution_param.pad_w


        if len(convolution_param.stride):
            self.__stride_h = convolution_param.stride[0]
            self.__stride_w = convolution_param.stride[0]

    def forward(self,blobs):
        print("%s forward"%self._name)
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


