from django.shortcuts import render
import sys


# from buttonpython.handtracking_ocr import MathewApp
from django.contrib import messages
from subprocess import run,PIPE
def home(request):
    return render(request,'home.html')
def improvements(request):
    return render(request,'improvements.html')
    
def ourTeam(request):
    return render(request,'ourTeam.html')

def about(request):
    return render(request,'about.html')
def modification(request):
    return render(request,'modification.html')

#change of pth using os 
def mlModel(request):
    run([sys.executable,'website/handTrackingModel.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')

def ocrModel(request):
    run([sys.executable,'website/handTrackingOcr.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')
