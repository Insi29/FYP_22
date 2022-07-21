#Import Library
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from src.utils.txttspeech import ttsp
import ctypes
from src.utils.cv_utils import getIdxToCoordinates, rescaleFrame
from src.utils.ocr import ocr
from src.utils.wordRecognition import word_recog_predict

class airWriting:
    #Initialization
    def __init__(self,method):
        #Initialization of Media Pipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        #Capitalize the method Name
        self.method=method.upper() 

    #FullScreen Implementation
    def fullScreen(self,WINDOW_NAME):
           cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
           cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #Exit Screen Implementation
    def exitScreen(self):       
        self.hands.close()
        self.cap.release()
    
    #Hand Tracking and Writing Event Implementation
    def handTracking(self):
        #Hand Detection using Media Pipe
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        hand_landmark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
        hand_connection_drawing_spec = self.mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
        WINDOW_NAME = 'Air Writing'
        self.cap = cv2.VideoCapture(0)
        #Full Screen Implementation
        self.fullScreen(WINDOW_NAME)
        pts = deque(maxlen=512)
        blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)+255
        break_taken = False
        img_counter=0
        frame_count = 0
        word_predicted=""
        if not self.cap.isOpened():
            #Raises IO Error 
            raise IOError("Cannot open webcam!")
        while self.cap.isOpened():
            idx_to_coordinates = {}
            _, image = self.cap.read()
            self.image=image
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            results_hand = self.hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_drawing_spec,
                        connection_drawing_spec=hand_connection_drawing_spec)
                    idx_to_coordinates = getIdxToCoordinates(image, results_hand)
            if 8 in idx_to_coordinates and 17 in idx_to_coordinates and idx_to_coordinates[17][0] > \
                    idx_to_coordinates[8][0]:    
                frame_count = 0
                pts.appendleft(idx_to_coordinates[8])  # Index Finger
            if break_taken == True and len(pts) > 0:
                #Condition for not Writing
                pts.appendleft(-1)   
                pts.appendleft(pts[0])
                break_taken = False
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None or pts[i] == -1 or pts[i - 1] == -1:
                    continue
                #print('Points:',pts) #Printing points
                #Writing Event
                #When pts>0 writing takes place
                cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), 8) 
                #Thickness of line drawn =8
                cv2.line(blackboard, pts[i - 1], pts[i], (0,0,0), 8) 
                #Thickness of line drawn =8

            if 8 not in idx_to_coordinates or 17 not in idx_to_coordinates or idx_to_coordinates[17][0] <= \
                    idx_to_coordinates[8][0]: 
                frame_count += 1
                break_taken = True      
                if len(pts) != [] and frame_count >= 50:   
                    #pts is not None that means user has taken break but still writes
                    break_taken = False
                    blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_RGB2GRAY)
                    blur1 = cv2.medianBlur(blackboard_gray, 15)
                    #Gaussain Blur smoothens the image
                    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                    #Thresh_binary necessary for binary image
                    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    #Finding Contours 
                    blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if blackboard_cnts is None:
                        frame_count = 0
                        continue
                    if len(blackboard_cnts) >= 1:
                        for cnt in blackboard_cnts:
                            if cv2.contourArea(cnt) > 1000:
                                 frame_count = 0
                    #Initializing Air Writing Screen
                    pts = deque(maxlen=512)
                    blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)+255
                    word_predicted=""
            
            k=cv2.waitKey(5)
            #ESC pressed to exit the screen the screen
            if k & 0xFF == 27: 
                break
            #SPACE pressed to capture frames and predict word
            if k%256 == 32:
                img_name = "opencv_frame{}.png".format(img_counter)
                cv2.imwrite(img_name, blackboard)
                img_counter += 1
                if self.method == "OCR":
                    #OCR prediction of text
                    prediction=ocr(img_name)
                    if prediction is None:
                        result='Try Again'
                    else:
                        result=str(prediction)
                    word_predicted+="Text="+result
                    #Text to Speech
                    ttsp(result)
                elif self.method == "ML":
                    #ML prediction of text
                    prediction=word_recog_predict(img_name)
                    word_predicted+="Text="+prediction
                    #Text to Speech
                    ttsp(prediction)
                else:
                    self.method="OCR"
            #Display Word predicted on the Screen
            image = cv2.putText(image, word_predicted ,  (200, 90),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #Display Method on the Screen
            image = cv2.putText(image, self.method, (400, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            cv2.imshow(WINDOW_NAME,image)
            cv2.imshow("BB", rescaleFrame(blackboard, percent=100))
           
        #Commands to exit the screen
        self.exitScreen()

