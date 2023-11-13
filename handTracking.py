import cv2 as cv
import mediapipe as mp
import cvzone

class HandTracking():

    def __init__(self):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.draw = mp.solutions.drawing_utils
        self.results = None

    def handDetect(self, img):

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                self.draw.draw_landmarks(img, hand_landmark, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findHand(self, img, handNo = 0):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                cv.circle(img, (cx, cy), 5, (255, 0, 0), 3, cv.FILLED)
                lmList.append((id, cx, cy))
        return lmList
    
    def badFinger(self, img, lmList):

        ids = [8, 12, 16, 20]
        fingers = []
        if len(lmList) >= max(ids):
            if lmList[4][1] < lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(0, 4):
                if lmList[ids[id]][2] < lmList[ids[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        if len(fingers) == 5 and sum(fingers) == 1 and fingers[2] == 1:
            h, w, c = img.shape
            lm = lmList[12]
            cx, cy = int(w * lm[1]), int(h * lm[2])
            cvzone.putTextRect(img, "Middle Finger Detected!!!!", (50, 50), scale =2, font=cv.FONT_HERSHEY_PLAIN)
        print(fingers)
        return img
