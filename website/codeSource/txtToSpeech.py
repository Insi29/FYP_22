#Import Library
import pyttsx3

def ttsp(txt):
    #Initialize 
    engine = pyttsx3.init()
    #Converts text to Speech
    engine.say("The predicted word is")
    engine.say(txt)
    engine.say("Thank you")
    engine.runAndWait()

