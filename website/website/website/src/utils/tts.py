import playsound
# Import the required module for text
# to speech conversion
from gtts import gTTS

def ttp(txt):
    # Language in which you want to convert
    language = 'en'
    # Passing the text and language to the engine,
    # # here we have marked slow=False. Which tells
    # # the module that the converted audio should
    # # have a high speed
    myobj = gTTS(text=txt, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")
    # Playing the converted file
    playsound.playsound('welcome.mp3')
# ttp('hi i am saima')