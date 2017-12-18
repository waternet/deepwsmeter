import cv2
import imutils
import glob

OUT_DIR = 'data/getallen/'

i = 1
for f in glob.glob('data/out*.jpg'):
    img = cv2.imread(f)
    img = imutils.rotate(img, 134)
    r1 = [225, 248, 623, 640]
    r2 = [225, 248, 644, 663]
    r3 = [224, 247, 667, 686]
    r4 = [224, 247, 692, 709]
    r5 = [223, 246, 714, 732]
    r6 = [225, 246, 739, 754]
    r7 = [224, 247, 762, 778]
    r8 = [224, 247, 787, 802]
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r1[0]:r1[1], r1[2]:r1[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r2[0]:r2[1], r2[2]:r2[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r3[0]:r3[1], r3[2]:r3[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r4[0]:r4[1], r4[2]:r4[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r5[0]:r5[1], r5[2]:r5[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r6[0]:r6[1], r6[2]:r6[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r7[0]:r7[1], r7[2]:r7[3]])
    i += 1
    cv2.imwrite(OUT_DIR + 'getal%05d.jpg' % i, img[r8[0]:r8[1], r8[2]:r8[3]])
    i += 1
