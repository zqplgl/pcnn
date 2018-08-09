import numpy as np
import sys
sys.path.append("..")
from layer import Layer
from proto import caffe_pb2

class ConvolutionLayer(Layer):
    def __init__(self,parameter,blobs):
        super(ConvolutionLayer, self).__init__(parameter)
        assert len(self._bottoms)==1,"%s bottom must equal 1: error %s"%(self._name,self._bottoms)
        assert self._bottoms[0] in blobs.keys(),"%s bottom %s is not init"%(self._name,self._bottoms[0])
        assert len(self._tops)==1,"%s top must equal 1: error %s"%(self._name,self._tops)
        assert self._tops[0] not in blobs.keys(),"%s top %s has been inited"%(self._name,self._tops[0])

        self.__extract_parameter(parameter.convolution_param)

        bottom_shape = blobs[self._bottoms[0]].shape
        assert len(bottom_shape)==4,"error bottom %s"%bottom_shape

        kernel_c = bottom_shape[1]
        self._w = np.zeros([self.__num_output,kernel_c,self.__kernel_h,self.__kernel_w],dtype=np.float32)
        self._b = np.zeros([self.__num_output],dtype=np.float32)

        top_shape = self.__generate_top_shape(bottom_shape)
        blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)

        sys.stderr.write("init Layer %s :%s->%s successfully\n"%(self._name,bottom_shape,blobs[self._tops[0]].shape))

    def __generate_top_shape(self,bottom_shape):
        out_n = bottom_shape[0]
        out_c = self.__num_output
        out_h = int((bottom_shape[2] + 2*self.__pad_h - self.__kernel_h)/self.__stride_h+1)
        out_w = int((bottom_shape[3] + 2*self.__pad_w - self.__kernel_w)/self.__stride_w+1)

        assert out_n>0 and out_c>0 and out_h>0 and out_w>0,"%s output feature map error:[%s,%s,%s,%s]"%(self._name,out_n,out_c,out_h,out_w)

        return (out_n,out_c,out_h,out_w)

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

    def __conv(self,im):
        assert len(im.shape)==2,"error conv blob"
        out_w = int((im.shape[1]+2*self.__pad_w-self.__kernel_w)/self.__stride_w +1)
        out_h = int((im.shape[0]+2*self.__pad_h-self.__kernel_h)/self.__stride_h +1)
        assert out_h>0 and out_w>0,"output feature map error:[%s,%s]"%(out_h,out_w)
        output = np.zeros([out_h,out_w],dtype=np.float32)

        for h in range(out_h):
            for w in range(out_w):
                for row in range(self.__kernel_h):
                    offset_h = self.__stride_h*h-self.__pad_h + row
                    if offset_h < 0 or offset_h >= im.shape[0]:
                        continue
                    for col in range(self.__kernel_w):
                        offset_w = self.__stride_w*w-self.__pad_w+col
                        if offset_w < 0 or offset_w >= im.shape[1]:
                            continue
                        output[h][w] += self._w[row][col]*im[offset_h][offset_w]

        return output

    def __conv(self,bottom_blob,top_blob):
        bottom_n,bottom_c,bottom_h,bottom_w = bottom_blob.shape
        top_n,top_c,top_h,top_w = top_blob.shape
        for t_n in range(top_n):
            for t_c in range(top_c):
                for b_c in range(bottom_c):
                    for h in range(top_h):
                        for w in range(top_w):
                            for row in range(self.__kernel_h):
                                offset_h = self.__stride_h*h-self.__pad_h + row
                                if offset_h < 0 or offset_h >= bottom_h:
                                    continue
                                for col in range(self.__kernel_w):
                                    offset_w = self.__stride_w*w-self.__pad_w+col
                                    if offset_w<0 or offset_w>=bottom_w:
                                        continue
                                    top_blob[t_n][t_c][h][w] += bottom_blob[t_n][b_c][offset_h][offset_w]*self._w[t_c][b_c][row][col]

                            top_blob[t_n][t_c][h][w] += self._b[t_c]

    def forward(self,blobs):
        bottom_blob = blobs[self._bottoms[0]]
        top_blob = blobs[self._tops[0]]

        assert len(bottom_blob.shape)==4 and bottom_blob.shape[1]==self._w.shape[1],"%s error input blobs"%self._name
        top_shape = self.__generate_top_shape(bottom_blob.shape)

        if top_blob.shape != top_shape:
            blobs[self._tops[0]] = np.zeros(top_shape,dtype=np.float32)
            sys.stderr.write("reshape blob %s from %s-->%s\n"%(self._tops[0],top_blob.shape,top_shape))
            top_blob = blobs[self._tops[0]]

        self.__conv(bottom_blob,top_blob)
        sys.stderr.write("%s forward successfully\n"%self._name)

if __name__=="__main__":
    a = [1,2,3,4]
    b = np.array(a,dtype=np.float32)
    print(b)


