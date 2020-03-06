import cv2
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)
lower_blue = np.array([94, 80, 75])
upper_blue = np.array([126, 255, 255])

while True: #cap.isOpened():
	_, frame = cap.read()
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	blurred_mask = cv2.medianBlur(mask, 15)

	ret,thresh = cv2.threshold(blurred_mask,127,255,0)

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	res = cv2.bitwise_and(frame, frame, mask=blurred_mask)

	for c in contours:
		if cv2.contourArea(c) >= 150:
			M = cv2.moments(c)

			if M["m00"] != 0:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			else:
				cX, cY = 0, 0

			cv2.drawContours(frame, [c], 0, (0,255,0), 3)
			cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
	 

	cv2.imshow("frame", frame)
	cv2.imshow("res", res)
	#cv2.imshow("blur", blurred_mask)
	
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
#cap.release()
