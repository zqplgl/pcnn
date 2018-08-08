import numpy as np
import sys
class Layer(object):
    def __init__(self,parameter):
        self._name = parameter.name
        self._type = parameter.type
        self._bottoms = parameter.bottom
        self._tops = parameter.top
        self._w = None
        self._b = None

    def bottom(self):
        return self._bottoms

    def top(self):
        return self._tops

    def forward(self,blobs):
        pass

    def backward(self,blobs):
        pass

    def load_parameter(self,blobs):
        w = np.reshape(np.array(blobs[0].data,dtype=np.float32),self.blobshape_convert(blobs[0].shape))
        b = np.array(blobs[1].data)
        assert b.shape==self._b.shape,"Layer %s w cannot convert parameter %s to %s"%(self._name,self._b.shape,b.shape)
        assert w.shape==self._w.shape,"Layer %s b cannot convert parameter %s to %s"%(self._name,self._w.shape,w.shape)
        self._w = w
        self._b = b

        sys.stderr.write("init Layer %s parameter w: %s b: %s successfully\n"%(self._name,self._w.shape,self._b.shape))

    def parameters(self):
        return self._w,self._b

    def blobshape_convert(self,blobshape):
        return [dim for dim in blobshape.dim]

if __name__=="__main__":
    pass
