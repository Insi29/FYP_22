import pyttsx3
def ttsp(txt):
    engine = pyttsx3.init()
    # testing
    engine.say("The predicted word is")
    engine.say(txt)
    engine.say("Thank you")
    engine.runAndWait()

