import cv2
import time
import mediapipe as mp

class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands = 1, min_detection_confidence = 0.85)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw = True):
        '''
        This method detects hand landmarks.
        '''
        # img = cv2.flip(img, 1)
        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = self.hands.process(imgRGB) 

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        '''
        This method finds the position of particular landmark and returns them as a list
        '''
        self.lmList = []  # List containing all landmark  

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         # Process the image to detect hand landmarks
        results = self.hands.process(imgRGB) 

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                #print(id, lm) # Each ID has a corresponding hand landmark
                    h, w, c = img.shape # Height, Width and Channels
                    cx, cy = int(lm.x*w), int(lm.y*h) # Position
                    self.lmList.append([id, cx, cy])
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self, lmList):
        '''
        This method tells which of the fingers are up
        '''
        fingers = []
        
        # Check if lmList has been initialized
        if len(lmList) != 0:
            #Thumb
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
    
            #looping through other 4 fingers
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            fingers = [0, 0, 0, 0, 0]
        return fingers