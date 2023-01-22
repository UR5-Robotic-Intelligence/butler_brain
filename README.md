# owl_test

A Repo to test using ontologies like knowrob with deeplearning in robotics.

# Dependencies

```
openai (GPT-3) -> pip install openai
fuzzywuzzy (string similarity) -> pip install fuzzywuzzy
pip install torch torchvision
pip install transformers
pip install -U sentence-transformers
gTTs (google text to speech) -> pip install gTTS
pyyaml -> pip install pyyaml
rospkg -> pip install rospkg
pyaudio -> sudo apt-get install portaudio19-dev && pip install pyaudio
pip install --upgrade google-cloud-speech
pip install SpeechRecognition
pip install google-cloud-texttospeech
pip install torchtext
rosprolog
knowrob
mpg321 -> sudo apt install mpg321
```
# Google Cloud API

https://cloud.google.com/docs/authentication/provide-credentials-adc

get sevice account key json file and add its path to bashrc:

export GOOGLE_APPLICATION_CREDENTIALS="path/to/json"
