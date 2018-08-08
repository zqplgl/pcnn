import numpy as np
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
        pass

    def parameters(self):
        return self._w,self._b

    def blobshape_convert(self,blobshape):
        return [dim for dim in blobshape.dim]

if __name__=="__main__":
    pass
