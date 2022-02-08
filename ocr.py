import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#smooth image
img1 = cv2.imread("download.png")
text = pytesseract.image_to_string(img1)
print(text)


#handwrtten Images
print("Handwritten Image\n")
img = cv2.imread("handwritten2.jpg")
#Resize the image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
#Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Convert image to black and white (using adaptive threshold)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
text = pytesseract.image_to_string(adaptive_threshold)
print(text)

