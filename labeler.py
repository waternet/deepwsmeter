import glob
import cv2
import shutil

images = glob.glob('data/getallen/*.jpg')

keys = {156:1, 153:2, 155:3, 150:4, 157:5, 152:6, 149:7, 151:8, 154:9, 158:0, 159:99}

for f in images:
    img = cv2.imread(f)
    print(f.split('/')[-1])
    cv2.imshow('', img)
    k = cv2.waitKey(0)
    print(keys[k])

    shutil.move(f, 'data/getallen/%d/%s' % (keys[k], f.split('/')[-1]))
