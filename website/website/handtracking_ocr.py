from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from src.utils.txttspeech import ttsp


#from MathewModel import MathewModel
from src.utils.cv_utils import get_idx_to_coordinates, rescale_frame
from src.utils.ocr import ocr



class MathewApp:
    def __init__(self):
        #Initialization of Media Pipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
       # self.mathew_model = MathewModel()

    def run(self):
        frame_count = 0
        #res = 0
        abc=""
        #Hand Detection
        hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        hand_landmark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
        hand_connection_drawing_spec = self.mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
        cap = cv2.VideoCapture(0)
        pts = deque(maxlen=512)
        # blackboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)+255
        char = np.zeros((200, 200, 3), dtype=np.uint8)
        #pred_class = 0
        break_taken = False
        res_list = deque(maxlen=512)
        img_counter=0
        while cap.isOpened():
            idx_to_coordinates = {}
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results_hand = hands.process(image)
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
                    idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
                    #print('Cordinates:',idx_to_coordinates)
            if 8 in idx_to_coordinates and 17 in idx_to_coordinates and idx_to_coordinates[17][0] > \
                    idx_to_coordinates[8][0]:    
                frame_count = 0
                pts.appendleft(idx_to_coordinates[8])  # Index Finger
            if break_taken == True and len(pts) > 0:
                pts.appendleft(-1)   #agr pts mai >0 value hai aur break liya wa hai toh append -1 so that it shows that there is not writing
                pts.appendleft(pts[0])
                break_taken = False
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None or pts[i] == -1 or pts[i - 1] == -1:
                    continue
                #print('Points:',pts) #Printing points
                cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), 8) #when pts>0 writing takes place
                #Thickness of line drawn =4
                cv2.line(blackboard, pts[i - 1], pts[i], (0,0,0), 8) 
                #Thickness of line drawn =4

            if 8 not in idx_to_coordinates or 17 not in idx_to_coordinates or idx_to_coordinates[17][0] <= \
                    idx_to_coordinates[8][0]: 
                frame_count += 1
                break_taken = True      
                if len(pts) != [] and frame_count >= 50:   #pts is not None that means user has taken break but still writes
                    break_taken = False
                    blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_RGB2GRAY)
                    blur1 = cv2.medianBlur(blackboard_gray, 15)
                    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if blackboard_cnts is None:
                        frame_count = 0
                        continue
                    if len(blackboard_cnts) >= 1:
                        res_list = []
                        for cnt in blackboard_cnts:
                            if cv2.contourArea(cnt) > 1000:
                                 x, y, w, h = cv2.boundingRect(cnt)
                                 frame_count = 0
                                ###commented above lines for some check
                    pts = deque(maxlen=512)
                    # blackboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)+255
                    abc=""
            pos = 0
            #if len(res_list) > 0:
                #res = solve_eqn(res_list)
                #res_list.clear()
                #Captures images of frames, Modularized
            #bb_img_name=capture(blackboard,img_counter)
            
          
            k=cv2.waitKey(5)
            if k & 0xFF == 27: #esc pressed which closes the screen
                break
            elif k%256 == 32:
                #SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, blackboard)
                # print("{} written!".format(img_name))
                img_counter += 1
                
                #prediction of text
                res=ocr(img_name)
                if res is None:
                    result='Try Again'
                else:
                    result=str(res)
                abc+="Text="+result
                #Text to Speech
                ttsp(result)
                
                #return res
                
            image = cv2.putText(image, abc ,  (200, 90),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            image = cv2.putText(image, "ocr", (400, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("AirWriting", rescale_frame(image, percent=100))
            cv2.imshow("BB", rescale_frame(blackboard, percent=100))
           
           
        hands.close()
        cap.release()
a=MathewApp()
output=a.run()
