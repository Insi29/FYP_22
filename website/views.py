from django.shortcuts import render
import sys


# from buttonpython.handtracking_ocr import MathewApp
from django.contrib import messages
from subprocess import run,PIPE
def home(request):
    return render(request,'home.html')
    
def our_team(request):
    return render(request,'our_team.html')

def about(request):
    return render(request,'about.html')
#change of pth using os 
def ml_model(request):
    run([sys.executable,'website/HandTracking_model.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')

def ocr_model(request):
    run([sys.executable,'website/handtracking_ocr.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')
