import cv2

def capture(blackboard, img_counter):
    img_name = r"opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, blackboard)
    print("{} written!".format(img_name))
    img_counter += 1
    return img_name