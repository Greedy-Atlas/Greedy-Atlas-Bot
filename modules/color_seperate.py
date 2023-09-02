import cv2
import numpy as np

# Camera Setting
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Running part
while cv2.waitKey(33) < 0 :
  ret, frame = capture.read()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lvalue = np.array([42, 103, 40])    
  rvalue = np.array([61, 255, 255])

  mask_green = cv2.inRange(hsv, lvalue, rvalue)
  kernel = np.ones((7,7),np.uint8)

  mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
  mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

  seg_cone = cv2.bitwise_and(hsv, hsv, mask=mask_green)
  contours, hier = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  output = cv2.drawContours(seg_cone, contours, -1, (0,0,255), 3)

  height, width, channel = output.shape
  color_save = open('color.txt', 'w')
  for y in range(0, height):
    for x in range(0, width):
      b = output.item(y, x, 0)
      g = output.item(y, x, 1)
      r = output.item(y, x, 2)

      color_save.write(f'({x},{y}) : blue: {b}, green: {g}, red: {r} \n')
  color_save.close()

  cv2.imshow("VideoFrame", frame)
  cv2.imshow("HSV track Bar", output)