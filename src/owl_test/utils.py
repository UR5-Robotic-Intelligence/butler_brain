import os
from gtts import gTTS
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

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


if __name__ == '__main__':
  print(text_to_keywords("Q:Prepare a meal for dinner please:\n", verbose=True))