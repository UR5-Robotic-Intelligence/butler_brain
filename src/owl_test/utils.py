import os
from gtts import gTTS


def text_to_speech(text, filename="test.mp3", verbose=False):
  if verbose:
    print(text)
  tts = gTTS(text=text, lang='en', slow=False,)
  tts.save(filename)
  os.system("mpg321 -q " + filename)
  os.system("rm " + filename)