import os
from gtts import gTTS
import openai
import speech_recognition as sr
import pyttsx3
import fuzzywuzzy.fuzz as fuzz

openai.api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

text_to_keyword_prompt = """Q:Put lemon on water please:
v. put
1. lemon
p. on
2. water
Q:Prepare a meal for dinner please:
v. prepare
1. meal
2. dinner
Q:Make me orange juice please:
v. make
1. orange
2. juice
Q:Pour me some water:
v. pour
1. water
Q:Some lemonade will do wonders for me right now:
1. lemonade
Q:Life without meat is nothing:
1.meat
Q:I want to eat some burgers:
1. burger\n
Q:I would love it if you can prepare me a nice drink:
v. prepare
1. drink
Q:Do you have some falafel?
1. falafel\n"""

def speach_to_text(verbose=True):
 
  # Initialize the recognizer
  r = sr.Recognizer()
  
  while(1):   
    # Exception handling to handle
    # exceptions at the runtime
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            #listens for the user's input
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            if len(MyText) > 0:
              print(MyText)
              return MyText
            # SpeakText(MyText)
    except sr.RequestError as e:
      if verbose:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
      if verbose:
        print("unknown error occurred")

def text_to_speech(text, filename="test.mp3", verbose=False):
  if verbose:
    print(text)
  tts = gTTS(text=text, lang='en', slow=False,)
  tts.save(filename)
  os.system("mpg321 -q " + filename)
  os.system("rm " + filename)
  
def text_to_keywords(text, verbose=False):
  # print(text_to_keyword_prompt+text)
  response = openai.Completion.create(
    engine="code-cushman-001",
    prompt=text_to_keyword_prompt+"Q:" + text + ":\n",
    temperature=0,
    max_tokens=30,
    top_p=0.1,
    n=1,
    stream=False,
    logprobs=None,
    stop=["Q:"]
  )
  if verbose:
    print(response)
  return response["choices"][0]["text"]

def get_top_matching_candidate(candidate_list, match_string):
  top_ratio = 0
  top_candidate_index = 0
  for i, candidate in enumerate(candidate_list):
      ratio = fuzz.partial_ratio(candidate, match_string)
      if ratio > top_ratio:
          top_ratio = ratio
          top_candidate_index = i
  return top_candidate_index


if __name__ == '__main__':
  # print(text_to_keywords("Q:Prepare a meal for dinner please:\n", verbose=True))
  speach_to_text()