import cv2
import glob
from time import time_ns
import json
import os
from os import path
file_path="./settings.json"
with open(file_path,"rt") as f:
    settings = json.loads(f.read())

img_array = []
fn=[ x for x in glob.glob('./generated/'+settings['file_folder']+'/**/*.jpg',recursive=True)]
fn.sort()
#print(fn)
for filename in fn: 
    img = cv2.imread(filename)
    # height, width, layers = img.shape
    # size = (width,height)
    img_array.append(img)

size=[960,1568]
out = cv2.VideoWriter('./generated/'+settings["file_folder"]+'/final.mp4',cv2.VideoWriter_fourcc(*'mp4v'), settings['fps'], size)
 
for i in range(len(img_array)):
    for m in range(settings['repititions']):
        out.write(img_array[i])
out.release()
with open(file_path,"wt") as f:
    settings['file_folder']=str(time_ns())
    f.write(json.dumps(settings,indent=4))
