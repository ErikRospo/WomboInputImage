import cv2

paths=["frames/%04d.jpg"%i for i in range(0,10000)]
imgArray = []
print("Starting reading")
for filename in paths:
    imgArray.append(cv2.imread(filename))
        
print("done reading")
print("writing video")
out = cv2.VideoWriter("res2.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 15, [1024,575])
for i in range(len(imgArray)):
    out.write(imgArray[i])
out.release()
print("Done making video")