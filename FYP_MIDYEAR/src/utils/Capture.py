import cv2

def capture(blackboard, img_counter):
    k = cv2.waitKey(5)
    if k%256 == 32:
                #SPACE pressed
                img_name = "E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, blackboard)
                print("{} written!".format(img_name))
                img_counter += 1
                return img_name