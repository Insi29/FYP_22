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