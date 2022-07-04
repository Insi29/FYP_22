# import cv2
# import numpy as np
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# def ocr(img_name):
#     img=cv2.imread(img_name)
#     # 2. Resize the image
#     #img = cv2.resize(img, None, fx=0.5, fy=0.5)

#     # 3. Convert image to grayscale
#     #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     # 4. Convert image to black and white (using adaptive threshold)
#     #adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
#     #print('Converting to text')
#     text = pytesseract.image_to_string(img)
#     #cv2.waitKey(0)
#     return text
#     #cv2.imshow("gray", gray)
#     #cv2.imshow("adaptive th", adaptive_threshold)
    
#NANONETS..
import requests
import json
def ocr(filename):
    url = 'https://app.nanonets.com/api/v2/OCR/Model/866d66dc-69d6-433c-82c8-6123cb2db3b6/LabelFile/'
    data = {'file': open(filename, 'rb')}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth('NlE3KM3daTBYb6TrHNUOJdHLMe1tSNLB', ''), files=data)
    res=json.loads(response.text)
    if res is None:
        return None
    #Checked for value is whether None or list out of range in both cases should return txt as None.
    #Exception Handling
    try:
        txt=res["result"][0]["prediction"][0]["ocr_text"]
        return txt
    except IndexError:
        return None