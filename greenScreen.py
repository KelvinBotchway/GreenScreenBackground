import cv2
import numpy as np

image = cv2.imread("bg.jpeg")
import cv2
import numpy as np

image = cv2.imread("bg.jpeg")
frame = cv2.imread("greenscreen.jpg")

frame = cv2.resize(frame, (640, 480))
image = cv2.resize(image, (640, 480))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

u_green = np.array([104, 255, 255])  # Adjust the upper boundary for lighter shades of green
l_green = np.array([30, 30, 0])

mask = cv2.inRange(hsv, l_green, u_green)  # Apply the range in the HSV color space
res = cv2.bitwise_and(frame, frame, mask=mask)

f = frame - res
f = np.where(f == 0, image, f)

cv2.imshow("image", frame)
cv2.imshow("mask", f)

cv2.waitKey(0)
cv2.destroyAllWindows()


frame = cv2.resize(frame, (640, 480))
image = cv2.resize(image, (640, 480))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

u_green = np.array([104, 255, 255])  # Adjust the upper boundary for lighter shades of green
l_green = np.array([30, 30, 0])

mask = cv2.inRange(hsv, l_green, u_green)  # Apply the range in the HSV color space
res = cv2.bitwise_and(frame, frame, mask=mask)

f = frame - res
f = np.where(f == 0, image, f)

cv2.imshow("image", frame)
cv2.imshow("mask", f)

cv2.waitKey(0)
cv2.destroyAllWindows()
