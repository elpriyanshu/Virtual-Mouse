import cv2
import mediapipe as mp
import time
import math
import numpy as np
 
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): # mode = false means some times it will track and some times it will detect
        # based on the confidence level, if we won't make mode false, it will always detect which will make it very slow
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands  # hands is a class in mediapipe, we are creating an object of that class
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # drawing markers
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting to RGB because hands module takes RGB image
        self.results = self.hands.process(imgRGB)  # hands.process will process the image and will give us the results, it returns <class 'mediapipe.python.solution_base.SolutionOutputs'> ???
        # print(results.multi_hand_landmarks) # it will give us the landmarks of the hand
 
        if self.results.multi_hand_landmarks: # if there is a single hand or multiple hands
            for handLms in self.results.multi_hand_landmarks: # for each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS) # mediapipe method that draws 21 landmarks on the hand (3 on each fingure and one at every fingure joint and 2 on palm)
                                                                              #(self.mpHands.HAND_CONNECTIONS)draws connections between them
 
        return img # returning the image with the landmarks

    def findPosition(self, img, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0] # we are taking the first hand
            for id, lm in enumerate(myHand.landmark): # (lm)landmark information has x, y and z coordinates, and (id) id number for each landmark, in sequence
                                                      # enumerate uses counter as a key (it will use 0,1,2... as key) to store/traverse... , so we can use it to get the id number
                # print(id, lm)
                h, w, c = img.shape # height, width and channel of the image
                cx, cy = int(lm.x * w), int(lm.y * h) # cx and cy are the coordinates of the landmarks
                                                      # lm.x and lm.y are the coordinates of the landmarks in percentage, of the img size, we need to multiply them with the width and height of the image to get the actual coordinates in pixels.
                xList.append(cx)
                yList.append(cy) # appending the x,y coordinates of the landmarks to the list
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) # change circle size and color of landmarks
 
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
 
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
 
        return self.lmList, bbox
 
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
 
        # Fingers
        for id in range(1, 5):
 
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        # totalFingers = fingers.count(1)
 
        return fingers
 
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, img, [x1, y1, x2, y2, cx, cy]
 
 
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()