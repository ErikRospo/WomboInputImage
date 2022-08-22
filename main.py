# imports and basic notebook setup
from io import StringIO
import os
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import json
import caffe
import cv2

with open("settings.json","rt") as f:
    settings=json.load(f)
# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist
model_path = './models/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# setting up progress bar.
status={
    "topLevel":"",
    "octave":"",
    "innerLevel":""
}
# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".

model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)
def fancy_status():
    print("Outer Level: "+status["topLevel"]+" Octave: "+status["octave"]+" Inner Level: "+status["innerLevel"])
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        status["innerLevel"]="(?/?)"
        status["octave"]="({}/{})".format(octave+1,octave_n)
        fancy_status()
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in range(iter_n):
            status["innerLevel"]="({}/{})".format(i+1,iter_n)
            fancy_status()
            make_step(net, end=end, clip=clip, **step_params)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

os.system('clear')    
img = np.float32(PIL.Image.open(settings['dream']['inputPath']))
os.makedirs(os.path.split(settings["dream"]["output"]["images"]["path"])[0],exist_ok=True)
frame = img
frame_i = 0
h, w = frame.shape[:2]
s = 0.05 # scale coefficient
paths=[]
for i in range(int(settings['dream']['iterations'])):
    status["topLevel"]="({}/{})".format(i+1,settings['dream']['iterations'])
    status["octave"]="(?/?)"
    status["innerLevel"]="(?/?)"
    fancy_status()
    frame = deepdream(net, frame,iter_n=int(settings['dream']['iterationsPer']),octave_n=int(settings['dream']['octaves']),octave_scale=float(settings['dream']['octaveScale']))
    fp=settings["dream"]["output"]["images"]["path"]%frame_i
    PIL.Image.fromarray(np.uint8(frame)).save(fp)
    paths.append(fp)
    frame = nd.affine_transform(frame,
                               [float(settings["transform"]["affine"]["x_factor"]),
                                float(settings["transform"]["affine"]["y_factor"]),1],
                           [h*s*float(settings["transform"]["affine"]["y_center"]),
                            w*s*float(settings["transform"]["affine"]["x_center"]),0], order=1)
    frame=nd.rotate(frame,float(settings["transform"]["rotate"]["angle"]))
    frame_i += 1
    

imgArray = []
print("Starting reading")
for filename in paths:
    imgArray.append(cv2.imread(filename))
        
print("done reading")
print("writing video")
out = cv2.VideoWriter(settings['dream']["output"]["video"]['path'],cv2.VideoWriter_fourcc(*settings['dream']["output"]["video"]['codec']),float(settings['dream']["output"]["video"]['fps']), [h,w])
for i in range(len(imgArray)):
    out.write(imgArray[i])
out.release()
print("Done making video")