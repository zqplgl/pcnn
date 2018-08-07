class Layer(object):
    def __init__(self,parameter):
        self._name = parameter.name
        self._type = parameter.type
        self._bottoms = parameter.bottom
        self._tops = parameter.top

    def bottom(self):
        return self._bottoms

    def top(self):
        return self._tops

    def forward(self,blobs):
        pass

    def backward(self,blobs):
        pass

if __name__=="__main__":
    pass
