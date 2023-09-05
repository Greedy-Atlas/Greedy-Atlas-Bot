import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480

# Camera Setting
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def trans_cor(percent):
  if(percent >= 50) :
    cor = -1 * ((50+(percent-100))/50)
  elif(percent < 50) :
    cor = (50-percent)/50
  else:
    cor = None
  return cor

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

  # Calc Moments
  avr_arr_cX = []
  avr_arr_cY = []

  output_dst = output.copy()
  gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
  ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

  contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

  for i in contours:
    M = cv2.moments(i)
    if(M['m00'] == 0): continue
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    if(cX >= WIDTH/2-1 and cX <=WIDTH/2+1 and cY >= HEIGHT/2-1 and cY <= HEIGHT/2+1): continue
    avr_arr_cX.append(cX)
    avr_arr_cY.append(cY)
    cv2.circle(output_dst, (cX, cY), 3, (255, 100, 0), -1)

  x_cor = trans_cor((np.mean(avr_arr_cX)/WIDTH)*100)
  y_cor = trans_cor((np.mean(avr_arr_cY)/HEIGHT)*100)

  # print(x_cor, y_cor)

  frame = cv2.flip(frame, 1)
  output_dst = cv2.flip(output_dst, 1)
  cv2.imshow("VideoFrame", frame)
  cv2.imshow("HSV track Bar", output_dst)