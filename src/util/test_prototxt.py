from google.protobuf import text_format
import caffe_pb2

if __name__=="__main__":
    prototxt = "/home/zqp/github/caffe/examples/mnist/lenet_train_test.prototxt"
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(),net)
    for layer in net.layer:
        if layer.type=="Convolution":
            print(layer)




    print("hello world")