class layer(object):
    def __init__(self,parameter):
        self._name = parameter.name
        self._type = parameter.type
        self._bottom = parameter.bottom
        self._top = parameter.top


    def forward(self,bottom=None,top=None):
        pass

    def backward(self,bottom=None,top=None):
        pass

if __name__=="__main__":
    pass
