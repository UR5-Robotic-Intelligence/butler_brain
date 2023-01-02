from __future__ import division

import os
from gtts import gTTS
import openai
import speech_recognition as sr
import fuzzywuzzy.fuzz as fuzz

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

import numpy as np
from tqdm import tqdm


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
1. burger\n
Q:I would love it if you can prepare me a nice drink:
1. drink
Q:Do you have some falafel?
1. falafel\n"""

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


def speach_to_text(verbose=True, show_all=False): # under development
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
    while(1):   
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

def get_top_matching_candidate(candidate_list, match_with_list):
  top_ratio = 0
  top_candidate_index = 0
  top_match_index = 0
  all_matches = []
  for i, candidate in enumerate(candidate_list):
    for j, match in enumerate(match_with_list):
      c, m = candidate.lower(), match.lower()
      ratio = fuzz.ratio(c, m) + fuzz.partial_ratio(c, m)
      ratio = ratio + 100*min(len(m), len(c)) if (c in m) or (m in c) else ratio
      all_matches.append((c, m, ratio))
      if ratio >= top_ratio:
          top_ratio = ratio
          top_candidate_index = i
          top_match_index = j
  print(all_matches)
  return top_candidate_index, top_match_index, top_ratio

def cos_sim(a, b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

if __name__ == '__main__':
  # print(text_to_keywords("Q:Prepare a meal for dinner please:\n", verbose=True))
  speach_to_text()
  # speach_to_text_()
  # data = {'alternative': [{'transcript': 't', 'confidence': 0.29859412}, {'transcript': 'tea'}, {'transcript': 'teeth'}, {'transcript': 'tee'}], 'final': True}
  # candidates = [val['transcript'] for val in data['alternative']]
  # match_with = ['MakingTea-TheBeverage', 'MakingCoffee-TheBeverage', 'MakingJuice']
  # candidate_index, match_index = get_top_matching_candidate(candidates, match_with)
  # print(candidates[candidate_index], match_with[match_index])