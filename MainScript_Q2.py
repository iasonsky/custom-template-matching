# Needed imports

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


img = cv2.imread('InputImages/jets_new.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
templ = cv2.imread('InputImages/jet_templ_new.jpg')
templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

plt.figure(dpi=300)
plt.subplot(121)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(122)
plt.axis("off")
plt.imshow(cv2.cvtColor(templ, cv2.COLOR_BGR2RGB))
plt.title('Template for matching')
plt.savefig("OutputImages/Q2/template_matching.jpg")
plt.show()

#Calculate the histogram of the template image
templ_hist = cv2.calcHist([templ], [0, 1, 2], None, [4,4,4],
                        [0, 256, 0, 256, 0, 256])
templ_hist = cv2.normalize(templ_hist, templ_hist).flatten()

(winW, winH) = (templ.shape[1], templ.shape[0])

result = img.copy()
counter = 0
# loop over the sliding window for each layer of the pyramid
for (x, y, window) in sliding_window(img, stepSize=24, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    # WINDOW
    # since we do not have a classifier, we'll just draw the window
    # clone = img.copy()
    # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    # cv2.waitKey(1)
    # time.sleep(0.01)

    #Calculate the histogram of the current window
    hist = cv2.calcHist([window], [0, 1, 2], None, [4, 4, 4],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    #Compare the histogram of the current window with the template histogram
    c = cv2.compareHist(templ_hist, hist, cv2.HISTCMP_CORREL)

    if(c > 0.992):
        # print(c)
        counter+=1
        cv2.rectangle(result, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

plt.figure(dpi=300)
plt.subplot(211)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(212)
plt.axis("off")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Identified: '+ str(counter))
plt.savefig("OutputImages/Q2/result_corr.jpg")
plt.show()


#Do the same for another similarity metric

result = img.copy()
counter = 0
# loop over the sliding window for each layer of the pyramid
for (x, y, window) in sliding_window(img, stepSize=24, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    # WINDOW
    # since we do not have a classifier, we'll just draw the window
    # clone = img.copy()
    # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    # cv2.waitKey(1)
    # time.sleep(0.025)

    #Calculate the histogram of the current window
    hist = cv2.calcHist([window], [0, 1, 2], None, [4, 4, 4],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    #Compare the histogram of the current window with the template histogram
    c = cv2.compareHist(templ_hist, hist, cv2.HISTCMP_CHISQR)

    if(c < 0.12):
        # print(c)
        counter+=1
        cv2.rectangle(result, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

plt.figure(dpi=300)
plt.subplot(211)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(212)
plt.axis("off")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Identified: '+ str(counter))
plt.savefig("OutputImages/Q2/result_chi_square.jpg")
plt.show()

print("End of MainScript_Q2")
