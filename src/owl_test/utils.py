from __future__ import division

import os
from gtts import gTTS
import openai
import speech_recognition as sr
import fuzzywuzzy.fuzz as fuzz
from sentence_transformers import SentenceTransformer


import re
import sys

from google.cloud import speech, texttospeech

import pyaudio
from six.moves import queue

import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time
import rospy


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
1. burger
Q:I would love it if you can prepare me a nice drink:
v. prepare
1. drink
Q:Do you have some falafel?
1. falafel"""

text_to_keyword_prompt = """Q:Put lemon on water please:
1. lemon
2. water
Q:Prepare a meal for dinner please:
1. meal
2. dinner
Q:Make me orange juice please:
1. orange
2. juice
Q:Pour me some water:
1. water
Q:Some lemonade will do wonders for me right now:
1. lemonade
Q:Life without meat is nothing:
1.meat
Q:I want to eat some burgers:
1. burger
Q:I would love it if you can prepare me a nice drink:
1. drink
Q:Do you have some falafel?
1. falafel
Q:Make me chocolate cake:
1.chocolate-cake
Q:Make me a meat burger:
1.meat-burger
Q:Make me some rice with salad:
1.rice
2.salad
Q:Prepare me a tuna sandwich with some tomato
1.tuna-sandwich
2-tomato
Q:Make me a cup of lemonade
1.cup
2.lemonade
Q:Prepare me a big rice plate
1.rice
2.plate"""

text_to_commands_drink = """Q:you squeeze the orange in the cup and then you put water in the cup:
1. transport(orange, cup)
2. pour(water, cup)
container(cup)
"""

text_to_commands_food = """Q:you spread the nautella on the bread and then you put it on the plate:
1. transport(nautella, plate)
2. transport(bread, plate)
container(plate)
"""
# Q: First put cheese on the bread, then put in the oven to heat it up for a few minutes, then take it out of the oven and enjoy:
# 1. transport(cheese, bread)
# 2. transport(bread, oven)
# 2. transport(bread, table)
# container(oven)
request_to_commands_drink = """Q:Make me orange juice please:
1. transport(orange, cup)
2. pour(water, cup)
container(cup)
"""

request_to_commands_food = """Q:Make me nautella sandwich please:
1. transport(nautella, plate)
2. transport(bread, plate)
container(plate)
"""

text_and_request_to_commands_drink = """Q:Make me orange juice please; To do it; you squeeze the orange in the cup and then you put water in the cup:
1. transport(orange, cup)
2. pour(water, cup)
container(cup)
"""

text_and_request_to_commands_food = """Q:Make me nautella sandwich please; To do it; you spread the nautella on the bread and then you put it on the plate:
1. transport(nautella, bread)
2. transport(bread, plate)
container(plate)
"""

# text_to_commands = """Q:you put tea packet in a cup and then you put water in the cup:
# 1. tea-packet
# 2. water
# 3. cup
# Q: First put cheese on the bread, then put in the oven to heat it up for a few minutes, then take it out of the oven and enjoy:
# 1. cheese
# 2. bread
# 3.
# Q: put oats in a bowl, add milk, add honey, mix it all together, and enjoy:
# 1. transport(oats, bowl)
# 2. pour(milk, bowl)
# 3. pour(honey, bowl)
# """

ont_to_commands_drink = """Q: {'output': 'Tea-Beverage', 'sim': [0.9059747, 0.8054108], 'objectActedOn': ['DrinkingMug', 'TeaPacket', 'Water'], 'level': 'activity', 'components': ['tea', 'beverage'], 'votes': 2, 'super_activities': ['PreparingABeverage'], 'super_objects': ['InfusionDrink'], 'score': 0.8556927442550659, 'name': 'MakingTea-TheBeverage', 'type': 'Drink', 'objects_details': {'DrinkingMug': [('type', 'Class'), ('subClassOf', 'Cup')], 'TeaPacket': [('type', 'Class'), ('subClassOf', 'DrinkingIngredient')], 'Water': [('type', 'Class'), ('subClassOf', 'ColorlessThing'), ('subClassOf', 'Drink'), ('subClassOf', 'DrinkingIngredient'), ('subClassOf', 'EnduringThing-Localized')]}}:
1. transport(tea-packet, drinking-mug)
2. pour(water, drinking-mug)
container(drinking-mug)
Q:{'output': 'Juice', 'sim': [0.9999999], 'objectActedOn': ['DrinkingGlass', 'Juice'], 'level': 'activity', 'components': ['juice'], 'votes': 1, 'super_activities': ['PreparingABeverage'], 'super_objects': ['Drink'], 'score': 0.9999998807907104, 'name': 'MakingJuice', 'type': 'Drink', 'objects_details': {'DrinkingGlass': [('type', 'Class'), ('subClassOf', 'DrinkingVessel')]}}:
1. pour(juice, drinking-glass)
container(drinking-glass)
"""

ont_to_commands_food = """
"""


components_to_steps_prompt = f"""Q:Components for making a tea-beverage are:
1. tea-packet : affords being diffused in water, being contained, being transported, is solid.
2. water : affords being contained, diffusing, dissolving, drinking, pouring, soaking, boiling, and washing, is liquid.
3. drinking-mug : affords containing, drinking, pouring, being transported, and washing, is solid.
steps:
1. transport(tea-packet, drinking-mug)
2. pour(water, drinking-mug)
Q:Components for making an egg-omlette are:
1. eggs : affords being broken, cooking, eating, being fried, being boiled, is solid.
2. oil : affords cooking, pouring, frying, being heated, being contained, is liquid.
3. pan : affords containing, pouring, cooking, frying, heating, being transported, and washing, is solid.
4. stove : affords cooking, heating, being turned-on, and being turned-off, is solid.
steps:
1. turn_on(stove)
2. transport(pan, stove)
3. pour(oil, pan)
4. break(eggs, pan)
5. wait(3, minutes)
6. turn_off(stove)
"""

prompts = {'text_to_keywords': text_to_keyword_prompt,
           'text_to_commands_drink': text_to_commands_drink,
           'ont_to_commands_food': ont_to_commands_food,
           'text_to_commands_food': text_to_commands_food,
           'ont_to_commands_drink': ont_to_commands_drink,
           'text_to_commands': text_to_commands_drink+text_to_commands_food,
           'ont_to_commands': ont_to_commands_food+ont_to_commands_drink,
           'components_to_steps': components_to_steps_prompt,
           'request_to_commands': request_to_commands_drink+request_to_commands_food,
           'request_to_commands_drink': request_to_commands_drink,
           'request_to_commands_food': request_to_commands_food,
           'text_and_request_to_commands_drink': text_and_request_to_commands_drink,
           'text_and_request_to_commands_food': text_and_request_to_commands_food,
           'text_and_request_to_commands': text_and_request_to_commands_drink+text_and_request_to_commands_food}

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

def write_embeddings(path, embeddings, vocab):
    
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            word = vocab[i]
            #skip words with unicode symbols
            # if len(word) != len(word.encode()):
            #     continue
            vector = ' '.join([str(i) for i in embedding.tolist()])
            f.write(f'{word} {vector}\n')


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue
        
        # return result
        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        if not result.is_final:
            # sys.stdout.write(transcript + overwrite_chars + "\r")
            # sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            num_chars_printed = 0
            return transcript + overwrite_chars
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0


def speach_to_text(verbose=True, show_all=False, stop_cond=None): # under development
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    if stop_cond is None:
      stop_cond = lambda : True
    while(stop_cond()):
      print("s2t stop cond = ", stop_cond())
      # Exception handling to handle
      # exceptions at the runtime
      try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            data =  listen_print_loop(responses)
            MyText = data.lower()
            if verbose:
              print(MyText)
            return MyText
            if len(data.alternatives[0].transcript) > 0:
              if verbose:
                print(data)
              if data['alternative'][0]['confidence'] >= 0.8 and not show_all:
                  MyText = data['alternative'][0]['transcript'].lower()
              elif show_all:
                MyText = [val['transcript'].lower() for val in data['alternative']]
              else:
                continue
              if verbose:
                print(MyText)
              return MyText
      except Exception as e:
        if verbose:
          print("unknown error occurred")

def speach_to_text_(verbose=True, show_all=False):
 
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
            data = r.recognize_google(audio2, show_all=True, language="en-US")
            # print(data)
            # MyText = MyText.lower()
            if len(data) > 0:
              if verbose:
                print(data)
              if data['alternative'][0]['confidence'] >= 0.8 and not show_all:
                  MyText = data['alternative'][0]['transcript'].lower()
              elif show_all:
                MyText = [val['transcript'].lower() for val in data['alternative']]
              else:
                continue
              if verbose:
                print(MyText)
              return MyText
            # SpeakText(MyText)
    except sr.RequestError as e:
      if verbose:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
      if verbose:
        print("unknown error occurred")

def text_to_speech_(text, filename="test.mp3", verbose=False):
  if verbose:
    print(text)
  tts = gTTS(text=text, lang='en', slow=False,)
  tts.save(filename)
  os.system("mpg321 -q " + filename)
  os.system("rm " + filename)

def text_to_speech(text, verbose=False):
    """Synthesizes speech from the input string of text."""
    if verbose:
      print(text)
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-E",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        volume_gain_db=6.0,
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        # print('Audio content written to file "output.mp3"')
    os.system("mpg321 -q " + "output.mp3")
    os.system("rm " + "output.mp3")
  
def tell_me_one_of(question, answers=['yes', 'no'], verbose=False, loop_stop_cond=None):
  answers = [ans.lower() for ans in answers]
  # if verbose:
  #   print(question)
  if loop_stop_cond is None:
    loop_stop_cond = lambda: True
    
  if type(loop_stop_cond) is int:
    start_time = time.time()
    wait_time = loop_stop_cond
    loop_stop_cond = lambda: not((time.time() - start_time < wait_time) and (not rospy.is_shutdown()))
  text_to_speech(question, verbose=verbose)
  while not rospy.is_shutdown():
    txt = speach_to_text(verbose=verbose)
    if txt is None:
      continue
    for answer in answers:
      if answer in txt:
        if verbose:
          print(answer)
        return answer
    txt = "I didn't get that, please tell me one of the following: " + str(answers)
    text_to_speech(txt, verbose=verbose)
  
def gpt(text, prompt_to_use='text_to_keywords', new_prompt="", verbose=False):
  # print(text_to_keyword_prompt+text)
  prompt = prompts[prompt_to_use] + new_prompt + "\n"
  response = openai.Completion.create(
    engine="code-cushman-001",
    prompt=prompt+"Q:" + text + ":",
    temperature=0,
    max_tokens=70,
    top_p=0.1,
    n=1,
    stream=False,
    logprobs=None,
    stop=["Q:"]
  )
  if verbose:
    print("The used gpt prompt is:")
    print(prompt+"Q:" + text + ":")
    print("The gpt response is:")
    print(response)
  return response["choices"][0]["text"], prompt

def get_activity_steps(activity_name, activity_components_dict, verbose=False):
  # print(text_to_keyword_prompt+text)
  prompt = f"""Q:Components for making a {activity_name} are:\n"""
  comp_num = 1
  for comp_name, comp_details in activity_components_dict.items():
    if comp_details is not None:
      prompt += f"""{str(comp_num)}.{comp_name}: """
      for i, detail in enumerate(comp_details):
        if i != len(comp_details)-1:
          prompt += f"""{detail}, """
        else:
          prompt += f"""{detail}\n"""
    else:
      prompt += f"""{str(comp_num)}.{comp_name}\n"""
    comp_num += 1
  prompt += "Steps:"
  response = openai.Completion.create(
    engine="code-cushman-001",
    prompt=components_to_steps_prompt + prompt,
    temperature=0,
    max_tokens=70,
    top_p=0.1,
    n=1,
    stream=False,
    logprobs=None,
    stop=["Q:"]
  )
  if verbose:
    print(response)
  return response["choices"][0]["text"]

def get_top_matching_candidate(candidate_list, match_with_list, bert=False, bert_model=None, verbose=False, sentence=False):
  top_ratio = 0
  top_candidate_index = 0
  top_match_index = 0
  all_matches = []
  candidate_list = list(candidate_list) if type(candidate_list) == list else [candidate_list]
  match_with_list = list(match_with_list) if type(match_with_list) == list else [match_with_list]
  if bert:
    model = SentenceTransformer('bert-base-nli-mean-tokens') if bert_model is None else bert_model
  for i, candidate in enumerate(candidate_list):
    for j, match in enumerate(match_with_list):
      c_m = [candidate, match]
      new_c_m = []
      for sent in c_m:
        sent = re.sub( r"([A-Z])", r" \1", sent).split()
        for w_idx, word in enumerate(sent):
          if word in [' ', '-', '_']:
            sent.remove(word)
          else:
            sent[w_idx] = word.strip('-').replace('-', ' ')
        sent = " ".join(sent)
        sent = sent.lower()
        new_c_m.append(sent)
      c, m = new_c_m[0], new_c_m[1]
      # c, m = candidate.lower(), match.lower()
      if bert:
        c_enc, m_enc = model.encode([c])[0], model.encode([m])[0]
        ratio = cos_sim(c_enc, m_enc)
      else:
        ratio = fuzz.ratio(c, m) + fuzz.partial_ratio(c, m)
        ratio = ratio + 100*min(len(m), len(c)) if (c in m) or (m in c) else ratio
      all_matches.append((c, m, ratio))
      if ratio >= top_ratio:
          top_ratio = ratio
          top_candidate_index = i
          top_match_index = j
  if verbose:
    print(all_matches)
  return top_candidate_index, top_match_index, top_ratio

def cos_sim(a, b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

if __name__ == '__main__':
#   print(gpt("""Components for making a coffee-beverage are:
# 1. coffee-powder : affords being diffused in water, being contained, being transported, is solid.
# 2. water : affords being contained, diffusing, dissolving, drinking, pouring, soaking, and washing, is liquid.
# 3. drinking-mug : affords containing, drinking, pouring, being transported, and washing, is solid.
# 4. sugar-cube: affords being disolved in water, being contained, being transported, is solid.
# steps""", verbose=True))
  # print(gpt("{'output': 'Coffee-Beverage', 'sim': [0.9146566], 'objectActedOn': ['CoffeeBeans', 'DrinkingMug', 'HotWater'], 'level': 'activity', 'components': ['coffee'], 'votes': 1, 'super_activities': ['PreparingABeverage'], 'super_objects': ['Drink'], 'score': 0.9146565794944763, 'name': 'MakingCoffee-TheBeverage', 'type': 'Drink', 'objects_details': {'CoffeeBeans': [('type', 'Class'), ('subClassOf', 'DrinkingIngredient'), ('subClassOf', 'Granular')], 'DrinkingMug': [('type', 'Class'), ('subClassOf', 'Cup')], 'HotWater': [('type', 'Class'), ('subClassOf', 'ColorlessThing'), ('subClassOf', 'Drink'), ('subClassOf', 'DrinkingIngredient'), ('subClassOf', 'EnduringThing-Localized')]}}", verbose=True))
  # print(gpt("{'output': 'Juice', 'sim': [0.9999999], 'objectActedOn': ['DrinkingGlass'], 'level': 'activity', 'components': ['juice'], 'votes': 1, 'super_activities': ['PreparingABeverage'], 'super_objects': ['Drink'], 'score': 0.9999998807907104, 'name': 'MakingJuice', 'type': 'Drink', 'objects_details': {'DrinkingGlass': [('type', 'Class'), ('subClassOf', 'DrinkingVessel')]}}", verbose=True))
  # res = get_activity_steps("coffee-beverage",
  #                          {"coffee-powder": ["affords being diffused in water",
  #                                             "being contained",
  #                                             "being transported",
  #                                             "is solid"],
  #                           "water": ["affords being contained",
  #                                     "diffusing",
  #                                     "dissolving",
  #                                     "drinking",
  #                                     "pouring",
  #                                     "soaking",
  #                                     "washing",
  #                                     "is liquid"],
  #                           "drinking-mug": ["affords containing",
  #                                            "drinking", "pouring",
  #                                            "being transported",
  #                                            "washing",
  #                                            "is solid"],
  #                           "sugar-cube": ["affords being disolved in water",
  #                                          "being contained",
  #                                          "being transported",
  #                                          "is solid"]}, verbose=True)
  # output_of_gpt3 = res.strip().strip().split("\n")
  gpt_string = gpt("{output': 'Tomatojuice', 'objectActedOn': ['tomato', 'cup', 'water', 'cup'], 'level': 'activity', 'type': 'Drink', 'container': 'cup', 'description': 'first put tomato in the cup and then pour water into the cup', 'components': ['in'], 'votes': 1, 'sim': [0.83004594], 'super_activities': ['PreparingABeverage'], 'super_objects': ['Drink'], 'objects_details': {'tomato': [], 'cup': [], 'water': []}}", prompt_to_use='ont_to_commands', verbose=True)
  print(gpt_string)
  gpt_string = gpt_string.strip().strip().split("\n")
  container = gpt_string[-1][10:-1]
  output_components = []
  rob_commands = []
  for i, step in enumerate(gpt_string):
    if i == len(gpt_string) - 1:
      break
    step = step.split(".")[-1]
    func_name = step.split("(")[0]
    input_args = step.split("(")[1].split(")")[0].split(",")
    input_args = [arg.strip() for arg in input_args]
    output_components.extend(input_args)
    rob_commands.append(input_args)
  print(rob_commands)
  # res = gpt("put tomato in cup, then pour water in cup", to_ont=True, verbose=True)
  # output_of_gpt3 = res.strip().strip().split("\n")
  # container = None
  # for i, step in enumerate(output_of_gpt3):
  #   if i == len(output_of_gpt3) - 1:
  #     container = step[10:-1]
  #     break
  #   step = step.split(".")[-1]
  #   func_name = step.split("(")[0]
  #   input_args = step.split("(")[1].split(")")[0].split(",")
  #   input_args = [arg.strip() for arg in input_args]
  #   print(func_name, input_args)
  # print(container)
  # print(gpt("Q:Prepare a meal for dinner please:\n", verbose=True))
  # speach_to_text()
  # speach_to_text_()
  # data = {'alternative': [{'transcript': 't', 'confidence': 0.29859412}, {'transcript': 'tea'}, {'transcript': 'teeth'}, {'transcript': 'tee'}], 'final': True}
  # candidates = [val['transcript'] for val in data['alternative']]
  # match_with = ['MakingTea-TheBeverage', 'MakingCoffee-TheBeverage', 'MakingJuice']
  # candidate_index, match_index = get_top_matching_candidate(candidates, match_with)
  # print(candidates[candidate_index], match_with[match_index])