from django.shortcuts import render
import sys


# from buttonpython.handtracking_ocr import MathewApp
from django.contrib import messages
from subprocess import run,PIPE
def home(request):
    return render(request,'home.html')
def improvements(request):
    return render(request,'improvements.html')
    
def our_team(request):
    return render(request,'our_team.html')

def lightslider(request):
    return render(request,'lightslider.js')


def JQuery(request):
    return render(request,'Jquery.js')

def script(request):
    return render(request,'script.js')

def about(request):
    return render(request,'about.html')
def modification(request):
    return render(request,'modification.html')

#change of pth using os 
def ml_model(request):
    run([sys.executable,'website/HandTracking_model.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')

def ocr_model(request):
    run([sys.executable,'website/handtracking_ocr.py'],shell=False,stdout=PIPE)
    return render(request,'home.html')
