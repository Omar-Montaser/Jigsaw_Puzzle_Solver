import cv2
import matplotlib.pyplot as plt

img = cv2.imread("0.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
