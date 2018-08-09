import caffe
import numpy as np
import cv2

def process_im():
    im_path = "../data/mnist/0/000003.png"
    im = cv2.imread(im_path,-1)
    im = im.astype(np.float32)
    im = im.reshape([1,1,28,28])

    return im

def run():
    prototxt = "../examples/mnist/deploy.prototxt"
    weight_file = "../examples/mnist/weight.caffemodel"

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt,weight_file,caffe.TEST)

    im = np.array([[[[1,2,3,4,5,6],[4,5,6,7,8,9],[1,2,3,4,5,6],[4,5,6,7,8,9],[1,2,3,4,5,6],[4,5,6,7,8,9]]]],dtype=np.float32)

    net.blobs["data"].reshape(*im.shape)
    net.blobs["data"].data[...] = im
    net.forward()
    print(net.blobs["conv1"].data)

def run_2():
    a = np.array([[1,2],[3,4]],dtype=np.float32)
    b = np.array([[1,2],[3,4]],dtype=np.float32)

    c = np.pad(a,2,"constant")
    print(c)

    print(a.shape)
    print(b.shape)

    c = np.dot(a,b)
    c = np.sum(a*b)
    print(c)

def conv(im,kernel):
    assert len(im.shape)==2,"im.shape must equal 2"
    assert len(kernel.shape)==2,"kernel.shape must equal 2"
    in_h,in_w = im.shape
    k_h,k_w = kernel.shape
    out_h = in_h-k_h+1
    out_w = in_w-k_w+1
    out = np.zeros((out_h,out_w),dtype=np.float32)

    for o_h in range(out_h):
        for o_w in range(out_w):
            for h in range(k_h):
                offset_h = o_h+h
                if offset_h>in_h:
                    continue
                for w in range(k_w):
                    offset_w = o_w+w
                    if offset_w>in_w:
                        continue
                    out[o_h][o_w] += im[offset_h][offset_w]*kernel[h][w]

    return out


def conv(im,kernel,pad,stride):
    output_w = int((im.shape[1]+2*pad-kernel.shape[1])/stride +1)
    output_h = int((im.shape[0]+2*pad-kernel.shape[0])/stride +1)
    output = np.zeros([output_h,output_w],dtype=np.float32)

    print("output_w: ",output_w)
    print("output_h: ",output_h)
    print("output: ",output.shape)

    anchor_center = [int(m/2) for m in kernel.shape]

    for h in range(output_h):
        for w in range(output_w):
            for row in range(kernel.shape[0]):
                offset_h = stride*h-pad + row
                if offset_h < 0 or offset_h >= im.shape[0]:
                    continue
                for col in range(kernel.shape[1]):
                    offset_w = stride*w-pad+col
                    if offset_w < 0 or offset_w >= im.shape[1]:
                        continue
                    output[h][w] += kernel[row][col]*im[offset_h][offset_w]


    print(output)

    print(anchor_center)

def run_3():
    kernel = np.array([[1,2,3],[4,5,6]],dtype=np.float32)
    kernel = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.float32)
    im = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,0]],dtype=np.float32)

    pad = 1
    stride = 1

    conv(im,kernel,pad,stride)

if __name__=="__main__":
    run()