# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import openai
import tiktoken
import tempfile
import IPython
import enum
import jonlog
import json
from gtts import gTTS
import uuid
import datetime as dt
import requests
import concurrent.futures
import base64
from github import Github
import time
import threading
import os
import random
import re
import io
import retrying
import pydub
from xml.dom import minidom
from xml.etree import ElementTree as ET
import requests
from bs4 import BeautifulSoup
import boto3
from botocore.exceptions import ClientError
import requests
import vertexai
import vertexai.preview.generative_models
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
import anthropic
import groq
import google.generativeai

try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

logger = jonlog.getLogger()
openai.api_key = os.environ.get("OPENAI_KEY", None) or open('/Users/jong/.openai_key').read().strip()
os.environ['GOOGLE_CLOUD_PROJECT'] = 'summer2023-392312'


# %%
class RateLimited:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.current_minute = time.strftime('%M')
        self.lock = threading.Lock()
        self.calls = 0

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            run = False
            with self.lock:
                current_minute = time.strftime('%M')
                if current_minute != self.current_minute:
                    self.current_minute = current_minute
                    self.calls = 0
                if self.calls < self.max_per_minute:
                    self.calls += 1
                    run = True
            if run:
                return fn(*args, **kwargs)
            else:
                time.sleep(15)
                return wrapper(*args, **kwargs)
                    
        return wrapper


# %%
class ElevenLabsTTS:
    WOMAN = 'EXAVITQu4vr4xnSDxMaL'
    MAN = 'VR6AewLTigWG4xSOukaG'
    BRIT_WOMAN = 'jnBYJClnH7m3ddnEXkeh'
    def __init__(self, voice_id=None):
        api_key_fpath='/Users/jong/.elevenlabs_apikey'
        with open(api_key_fpath) as f:
            self.api_key = f.read().strip()
        self._voice_id = voice_id or self.WOMAN
        self.uri = "https://api.elevenlabs.io/v1/text-to-speech/" + self._voice_id
        
    @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
    def tts(self, text):
        headers = {
            "accept": "audio/mpeg",
            "xi-api-key": self.api_key,
        }
        payload = {
            "text": text,
        }
        return requests.post(self.uri, headers=headers, json=payload).content


# %%
class GttsTTS:
    WOMAN = 'us'
    MAN   = 'co.in'
    def __init__(self, voice_id=None):
        self.tld = voice_id

    @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
    def tts(self, text):
        speech = gTTS(text=text, lang='en', tld=self.tld, slow=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_filename = f'{tmpdir}/audio'
            speech.save(temp_filename)
            with open(temp_filename, 'rb') as f:
                return f.read()


# %%
class OpenAITTS:
    """https://platform.openai.com/docs/guides/text-to-speech"""
    WOMAN = 'nova'
    MAN = 'echo'
    def __init__(self, voice_id=None, model='tts-1'):
        """Voices:
        alloy, echo, fable, onyx, nova, and shimmer
        Models:
        tts-1, tts-1-hd
        """
        self.voice = voice_id
        self.model = model

    @RateLimited(95)
    @jonlog.retry_with_logging()
    def tts(self, text):
        response = openai.OpenAI(api_key=openai.api_key).audio.speech.create(
          model=self.model,
          voice=self.voice,
          input=text
        )
        return response.content

    def tostring(self): return f"[OpenAITTS] {self.voice=} {self.model=}"
    def __repr__(self): return self.tostring()
    def __str__(self): return self.tostring()


# %%
class AWSPollyTTS:
    WOMAN = 'Kimberly'
    MAN = 'Matthew'
    BRIT_WOMAN = 'Amy'

    def __init__(self, voice_id=None):
        self.client = boto3.client('polly', region_name="us-east-1")
        self._voice_id = voice_id or self.WOMAN

    @RateLimited(95)
    @jonlog.retry_with_logging()
    def tts(self, text, ssml=False):
        kwargs = {}
        if ssml:
            kwargs['TextType'] = 'ssml'
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=self._voice_id,
            Engine="neural",
            **kwargs,
        )
        # The audio stream containing the synthesized speech
        audio_stream = response.get('AudioStream')
        return audio_stream.read()


# %%
from google.cloud import texttospeech_v1 as texttospeech

class GoogleTTS:
    WOMAN = 'en-US-Wavenet-F'
    MAN = 'en-US-Wavenet-D'
    BRIT_WOMAN = 'en-GB-Wavenet-A'

    def __init__(self, voice_name=None):
        self.client = texttospeech.TextToSpeechClient()
        self._voice_name = voice_name or self.WOMAN

    # Assuming the RateLimited and retry_with_logging decorators are defined elsewhere
    @RateLimited(95)
    @jonlog.retry_with_logging()
    def tts(self, text):
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=self._voice_name[:5],  # Extracts the language code from the voice name
                name=self._voice_name,
                # ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
        except:
            logger.critical(f"GoogleTTS error from {text=}")
            raise
        return response.audio_content

    @classmethod
    def list_voices(cls):
        data = texttospeech.TextToSpeechClient().list_voices()
        voices = [
            v.name for v in data.voices
            if v.name[:2] == 'en'
            and 'studio' not in v.name.lower()
            and 'journey' not in v.name.lower()
        ]
        return voices

    def tostring(self): return f"[GoogleTTS] {self._voice_name=}"
    def __repr__(self): return self.tostring()
    def __str__(self): return self.tostring()


# %%
def get_random_voices(n=2, openai=True, aws=True, google=True):
    possible = []
    if openai:
        possible += [OpenAITTS(voice_id=vid) for vid in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']]
    if aws:
        possible += [AWSPollyTTS(voice_id=vid) for vid in ['Kimberly', 'Matthew', 'Amy']]
    if google:
        possible += [GoogleTTS(vid) for vid in GoogleTTS.list_voices()]
    return random.sample(possible, n)


# %%
class AWSChat:
    MODELS = {
        "claude-instant": "anthropic.claude-instant-v1",
        "claude-best": "anthropic.claude-v2:1",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    }

    
    @classmethod
    def consolidate_messages(cls, message_list):
        if not message_list:
            return []
        consolidated = []
        current_role = None
        current_content = ""

        for message in message_list:
            role = message.get("role")
            content = message.get("content", "")
    
            if role == "system":
                role = "user"
            if role == current_role:
                current_content += "\n" + content
            else:
                if current_role is not None:
                    consolidated.append({"role": current_role, "content": current_content})
                current_content = content
                current_role = role
    
        if current_role is not None:
            consolidated.append({"role": current_role, "content": current_content})
    
        return consolidated

    @classmethod
    def msg(cls, messages=None, model="anthropic.claude-3-haiku-20240307-v1:0", **kwargs):
        client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        try:
            # The different model providers have individual request and response formats.
            # For the format, ranges, and default values for Anthropic Claude, refer to:
            # https://docs.anthropic.com/claude/reference/complete_post

            # Claude requires you to enclose the prompt as follows:
            # enclosed_prompt = "Human: " + prompt + "\n\nAssistant:"
            # prompt = "\n\n".join(
            #     [f'{"" if (msg["role"] == "system" and model != cls.MODELS["claude-instant"]) else ("Human" if msg["role"] != "assistant" else "Assistant")}: {msg["content"]}' for msg in messages] +
            #     ["Assistant:"]
            # )

            if 'temperature' not in kwargs:
                kwargs['temperature'] = 1
            system = "\n".join([m['content'] for m in messages if m['role'] == 'system'])
            other_msgs = cls.consolidate_messages([m for m in messages if m['role'] != 'system'])
            
            body = {
                "system": system,
                "messages": other_msgs,
                "max_tokens": 2048,
                "anthropic_version": "bedrock-2023-05-31",
                **kwargs
            }
            response = client.invoke_model(
                modelId=model, body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            completion = response_body["content"][0]["text"]
            return completion
        except ClientError as e:
            logger.exception(f"Couldn't invoke {model}", e)
            raise


# %%
class GoogleChat:
    MODELS = {
        "gemini-pro": "gemini-pro",
        "gemini-1.5-flash": "gemini-1.5-flash-latest",
    }

    @classmethod
    def get_apikey(cls):
        return os.environ.get("GEMINI_API_KEY") or open(os.path.expanduser("~/.google_apikey")).read().strip()

    @classmethod
    def consolidate_messages(cls, message_list):
        if not message_list:
            return []
        consolidated = []
        current_role = None
        current_content = ""

        for message in message_list:
            role = message.get("role")
            content = message.get("content", "")
    
            if role == "system":
                role = "user"
            if role == current_role:
                current_content += "\n" + content
            else:
                if current_role is not None:
                    consolidated.append({"role": current_role, "content": current_content})
                current_content = content
                current_role = role

        if current_role is not None:
            consolidated.append({"role": current_role, "content": current_content})
    
        return consolidated

    @classmethod
    def msg(cls, messages=None, model_name="gemini-1.5-flash-latest", **kwargs):
        google.generativeai.configure(api_key=cls.get_apikey())
        
        # Create the model configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        # generation_config.update(kwargs)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Initialize the model
        model = google.generativeai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        
        # Consolidate messages
        consolidated_messages = cls.consolidate_messages(messages)
        final_msg = "Continue"
        if consolidated_messages[-1]['role'] == 'user':
            final_msg = consolidated_messages.pop(-1)['content']
        # Create chat session history
        history = []
        for msg in consolidated_messages:
            history.append({
                "role": "user" if msg["role"] != "assistant" else "model",
                "parts": [msg["content"]]
            })
        # Start chat session
        chat_session = model.start_chat(history=history)
        # Send message and get response
        response = chat_session.send_message(final_msg)
        return response.text


# %%
class MistralChat:
    MODELS = {
        "mistral-medium": "mistral-medium",
        "mistral-small": "mistral-small",
    }

    @classmethod
    def get_apikey(cls):
        return os.environ.get("MISTRAL_API_KEY") or open('/Users/jong/.mistral_apikey').read().strip()

    @classmethod
    def consolidate_messages(cls, message_list):
        if not message_list:
            return []
    
        consolidated = []
        current_role = None
        current_content = ""
    
        for message in message_list:
            role = message["role"]
            content = message["content"]

            if role == current_role:
                current_content += "\n" + content
            else:
                if current_role is not None:
                    consolidated.append({"role": current_role, "content": current_content})
                current_content = content
                current_role = role
    
        if current_role is not None:
            consolidated.append({"role": current_role, "content": current_content})

        return consolidated

    @classmethod
    def msg(cls, messages=None, model="mistral-medium", **kwargs):
        client = MistralClient(api_key=cls.get_apikey())
        if not any(msg['role'] == 'user' for msg in messages):
            messages[-1]['role'] = 'user'
        chat_response = client.chat(
            model=model,
            messages=[MistralChatMessage(**msg) for msg in cls.consolidate_messages(messages)],
        )
        return chat_response.choices[0].message.content


# %%
class AnthropicChat:
    MODELS = {
        "claude-opus": "claude-3-opus-20240229",   # Large
        "claude-sonnet": "claude-3-sonnet-20240229", # Medium
        "claude-haiku": "claude-3-haiku-20240307",  # Small
    }

    @classmethod
    def get_apikey(cls):
        return os.environ.get("ANTHROPIC_APIKEY") or open('/Users/jong/.anthropic_apikey').read().strip()

    @classmethod
    def msg(cls, messages=None, model="claude-3-haiku-20240307", **kwargs):
        messages = MistralChat.consolidate_messages(messages)
        system = '\n'.join([msg['content'] for msg in messages if msg['role'] == 'system'])
        messages = [m for m in messages if m['role'] != 'system' and m['content']]
        if not messages:
            messages.append({'role': 'user', 'content': 'Continue.'})
        client = anthropic.Anthropic(api_key=cls.get_apikey())
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 4096
        message = client.messages.create(
            model=model,
            # max_tokens=4096,
            messages=messages,
            system=system,
            # temperature=0
            **kwargs,
        ).content[0].text
        return message


# %%
class GroqChat:
    MODELS = {
        "llama3": "Llama3-70b-8192",
        "mixtral": "Mixtral-8x7b-32768",
    }

    @classmethod
    def get_apikey(cls):
        return os.environ.get("GROQ_APIKEY") or open('/Users/jong/.groq_apikey').read().strip()

    @classmethod
    def msg(cls, messages=None, model="Mixtral-8x7b-32768", **kwargs):
        messages = MistralChat.consolidate_messages(messages)
        system = '\n'.join([msg['content'] for msg in messages if msg['role'] == 'system'])
        messages = [m for m in messages if m['role'] != 'system' and m['content']]
        if not messages:
            messages.append({'role': 'user', 'content': 'Continue.'})
        messages = [{'role': 'system', 'content': system}] + messages

        client = groq.Groq(api_key=cls.get_apikey())
        message = client.chat.completions.create(
            messages=messages,
            model=model,
        ).choices[0].message.content
        
        return message


# %%
class TogetherChat:
    MODELS = {
        "mixtral": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "qwen": "Qwen/Qwen1.5-110B-Chat",
        "databricks": "databricks/dbrx-instruct",
        "dolphin": "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    }

    @classmethod
    def get_apikey(cls):
        return api_key=os.environ.get("TOGETHER_API_KEY") or open(os.path.expanduser("~/.together_apikey")).read().strip()

    @classmethod
    def msg(cls, messages=None, model=None, **kwargs):
        if model is None:
            model = random.choice(cls.MODELS.values())
        # Prepare the messages for the API
        consolidated_messages = GoogleChat.consolidate_messages(messages)
        # Prepare the messages for the API
        api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in consolidated_messages]
        # Define the payload
        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 8000)
        }
        # Make the request to Together API
        headers = {
            "Authorization": f"Bearer {cls.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        # Parse the response
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]


# %%
# TogetherChat.msg([{'role': 'system', 'content': 'Give me a list of insane commercial genres'}])
# # !pip install -U together

# %%
# DEFAULT_MODEL = 'gpt-4-1106-preview'
# DEFAULT_LENGTH  = 80_000
DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_LENGTH  = 29_000

class Chat:
    class Model(enum.Enum):
        GPT3_5 = "gpt-3.5-turbo"
        GPT_4  = "gpt-4-turbo-preview"

    def __init__(self, system, max_length=DEFAULT_LENGTH, default_model=None, messages=None):
        self._system = system
        self._max_length = max_length
        self._default_model = default_model
        self._history = [
            {"role": "system", "content": self._system},
        ]
        if messages:
            self._history += messages

    @classmethod
    def num_tokens_from_text(cls, text, model=DEFAULT_MODEL):
        """Returns the number of tokens used by some text."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Lol openai probably the same
        return len(encoding.encode(text))
    
    @classmethod
    def num_tokens_from_messages(cls, messages, model=DEFAULT_MODEL):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Lol openai probably the same
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
    def _msg(self, *args, model=None, **kwargs):
        if model is None:
            if self._default_model is not None: model = self._default_model 
            else: model = DEFAULT_MODEL
        logger.info(f'requesting chatcompletion {model=}...')
        if model.startswith("AWS/"):
            model = model[4:]
            resp = AWSChat.msg(
                messages=self._history,
                **kwargs
            )
        elif model.startswith("GOOGLE/"):
            model = model[7:]
            resp = GoogleChat.msg(messages=self._history, model=model, **kwargs)
        elif model.startswith("MISTRAL/"):
            model = model[8:]
            resp = MistralChat.msg(messages=self._history, model=model, **kwargs)
        elif model.startswith("ANTHROPIC/"):
            model = model[10:]
            resp = AnthropicChat.msg(messages=self._history, model=model, **kwargs)
        elif model.startswith("GROQ/"):
            model = model[5:]
            resp = GroqChat.msg(messages=self._history, model=model, **kwargs)
        elif model.startswith("TOGETHER/"):
            model = model[9:]
            resp = TogetherChat.msg(messages=self._history, model=model, **kwargs)
        else:
            resp = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
                *args,
                model=model,
                messages=self._history,
                **kwargs
            ).choices[0].message.content
        logger.info(f'received chatcompletion {model=}...')
        return resp
    
    def message(self, next_msg=None, **kwargs):
        # TODO: Optimize this if slow through easy caching
        while len(self._history) > 1 and self.num_tokens_from_messages(self._history) > self._max_length:
            logger.info(f'Popping message: {self._history.pop(1)}')
        if next_msg is not None:
            self._history.append({"role": "user", "content": next_msg})
        logger.info(f'Currently at {self.num_tokens_from_messages(self._history)=} tokens in conversation')
        resp = self._msg(**kwargs)
        text = resp
        self._history.append({"role": "assistant", "content": text})
        return text


# %%
class PodcastChat(Chat):
    def __init__(self, topic, podcast="award winning", max_length=DEFAULT_LENGTH, hosts=['Tom', 'Jen'], host_voices=[AWSPollyTTS(AWSPollyTTS.MAN), OpenAITTS(OpenAITTS.WOMAN)], extra_system=None, system=None):
        if system is None:
            system = f"""You are an {podcast} podcast with hosts {hosts[0]} and {hosts[1]}.
Respond with the hosts names before each line like {hosts[0]}: and {hosts[1]}:""".replace("\n", " ")
        if extra_system is not None:
            system = '\n'.join([system, extra_system])
        super().__init__(system, max_length=max_length)
        self._podcast = podcast
        self._topic = topic
        self._hosts = hosts
        self._history.append({
            "role": "user", "content": f"""Generate an informative, entertaining, and very detailed podcast episode about {topic}.
Make sure to teach complex topics in an intuitive way.""".replace("\n", " ")
        })
        self._tts_h1, self._tts_h2 = host_voices

    def text2speech(self, text, spacing_ms=350):
        tmpdir = '/tmp'
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as thread_pool:
            i = 0
            jobs = []
            def write_audio(msg, i, voice, **kwargs):
                logger.info(f'requesting tts {i=} {voice=}')
                s = voice.tts(msg)
                logger.info(f'received tts {i=} {voice=}')
                return s

            text = text.replace('\n', '!!!LINEBREAK!!!').replace('\\', '').replace('"', '')
            # Build text one at a time
            currline, currname = "", self._hosts[0]
            name2tld = {self._hosts[0]: 'co.uk', self._hosts[1]: 'com'}
            name2voice = {self._hosts[0]: self._tts_h1, self._hosts[1]: self._tts_h2}
            audios = []
            for line in text.split("!!!LINEBREAK!!!"):
                if not line.strip(): continue
                if line.startswith(f"{self._hosts[0]}: ") or line.startswith(f"{self._hosts[1]}: "):
                    if currline:
                        jobs.append(thread_pool.submit(write_audio, currline, i, name2voice[currname], lang='en', tld=name2tld[currname]))
                        i += 1
                    currline = line[4:]
                    currname = line[:3]
                else:
                    currline += line
            if currline:
                jobs.append(thread_pool.submit(write_audio, currline, i, name2voice[currname], lang='en', tld=name2tld[currname]))
                i+=1
            # Concat files
            audios = [job.result() for job in jobs]
            logger.info('concatting audio')
            audio = merge_mp3s(audios)
            logger.info('done with audio!')
            IPython.display.display(IPython.display.Audio(audio, autoplay=False))
            return audio
            
    def step(self, msg=None, skip_aud=False, ret_aud=True, min_length=None, **kwargs):
        msg = self.message(msg, **kwargs)
        if min_length is not None and len(msg) < min_length:
            raise ValueError(f"Message [{msg}] is shorter than {min_length=}")
        if skip_aud: return msg
        aud = self.text2speech(msg)
        if ret_aud: return msg, aud
        return msg


# %%
class PodcastXMLHandler:
    def __init__(self):
        self.root = ET.Element("channel")  # 'channel' is typically used in podcast RSS feeds
        self.tree = ET.ElementTree(self.root)

    def to_xml(self, filepath):
        self.tree.write(filepath, encoding='utf-8', xml_declaration=True, pretty_print=True)

    @classmethod
    def from_xml(cls, filepath):
        self = cls()
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        return self

    def contains_episode(self, episode_name):
        for episode in self.root.findall('./channel/item'):
            title = episode.find('title').text
            if title == episode_name:
                return True
        return False

    def remove_episodes_older_than(self, limit):
        now = datetime.datetime.now()
        parent_map = {c:p for p in self.root.iter() for c in p}
        for episode in self.root.findall('./channel/item'):
            pub_date = datetime.datetime.strptime(episode.find('pubDate').text, '%a, %d %b %Y %H:%M:%S %Z')  # RSS date format
            if now - pub_date > limit:
                parent_map[episode].remove(episode)

    def add_episode(self, episode_details):
        episode = ET.SubElement(self.root, './channel/item')
        for key, value in episode_details.items():
            ET.SubElement(episode, key).text = str(value)

"""
pd = PodcastXMLHandler.from_xml('/Users/jong/Downloads/podcast.xml')
pd.contains_episode('cs.IR: Recent Research Papers on Data Science and Cybersecurity.')
pd.remove_episodes_older_than(datetime.timedelta(days=30))
pd.to_xml('/Users/jong/Downloads/podcast2.xml')
"""
pass


# %%
class PodcastRSSFeed:
    """Class to handle rss feed operations using github pages."""

    def __init__(self, org, repo, xml_path, clean_timedelta=None):
        self.org = org
        self.repo = repo
        self.xml_path = xml_path
        self.local_xml_path = self.download_podcast_xml()
        self.clean_timedelta = clean_timedelta

    def get_file_base64(self, file_path):
        with open(file_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def download_podcast_xml(self):
        outfile = tempfile.NamedTemporaryFile().name + '.xml'
        raw_url = f'https://raw.githubusercontent.com/{self.org}/{self.repo}/main/{self.xml_path}'
        response = requests.get(raw_url)
        print(raw_url)
        if response.status_code != 200:
            raise Exception(response.text)
        with open(outfile, 'wb') as file:
            file.write(response.content)
        return outfile

    def update_podcast_xml(self, xml_data, file_name, episode_title, episode_description, file_length):
        # Parse XML
        root = ET.fromstring(xml_data)
        channel = root.find('channel')

        file_extension = os.path.splitext(file_name)[-1].lower()[1:]
        content_type = 'audio/' + file_extension
        
        # Add new episode
        item = ET.SubElement(channel, 'item')
        ET.SubElement(item, 'title').text = episode_title
        ET.SubElement(item, 'description').text = episode_description
        ET.SubElement(item, 'pubDate').text = dt.datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
        ET.SubElement(item, 'enclosure', {
            'url': f'https://{self.org}.github.io/{file_name}',
            'type': content_type,
            'length': str(file_length),
        })
        ET.SubElement(item, 'guid').text = str(uuid.uuid4())

        # Convert back to string and pretty-format
        pretty_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
        # Remove extra newlines
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])

        return pretty_xml

    def remove_episodes_older_than(self, xml_data, limit):
        now = dt.datetime.now()
        root = ET.fromstring(xml_data)
        parent_map = {c:p for p in root.iter() for c in p}
        token = os.environ.get("GH_KEY", None) or open("/Users/jong/.gh_token").read().strip()
        gh = Github(token)
        made_changes = False
        for episode in root.findall('./channel/item'):
            pub_date = dt.datetime.strptime(episode.find('pubDate').text, '%a, %d %b %Y %H:%M:%S %Z')  # RSS date format
            if now - pub_date > limit:
                episode_path = episode.find('enclosure').attrib['url'].split('.github.io/', 1)[1]
                logger.info(f"Deleting old episode: {episode_path}")
                parent_map[episode].remove(episode)
                made_changes = True
                # Get the repository
                try:
                    repo = gh.get_user().get_repo(self.repo)
                except:
                    repo = gh.get_organization(self.org).get_repo(self.repo)
                try:
                    contents = repo.get_contents(episode_path)
                    repo.delete_file(episode_path, "remove due to date", contents.sha)
                except Exception as e:
                    logger.exception(e)
        # Convert back to string and pretty-format
        pretty_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
        # Remove extra newlines
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])
        # Upload
        if made_changes:
            try:
                podcast_xml_sha = repo.get_contents(self.xml_path).sha
                self.upload_to_github(self.xml_path, pretty_xml, f'Delete old episodes in podcast.xml', podcast_xml_sha)
            except Exception as e:
                logger.exception(e)
        return pretty_xml
    
    def upload_episode(self, file_path, file_name, episode_title, episode_description):
        # Authenticate with GitHub
        token = os.environ.get("GH_KEY", None) or open("/Users/jong/.gh_token").read().strip()
        gh = Github(token)

        # Get the repository
        try:
            repo = gh.get_user().get_repo(self.repo)
        except:
            repo = gh.get_organization(self.org).get_repo(self.repo)

        # Upload the audio file
        podsha = None
        try:
            podsha = repo.get_contents(file_name).sha
        except:
            pass
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            self.upload_to_github(file_name, audio_data, f'Upload new episode: {file_name}', podsha)

        # Update and upload the podcast.xml file
        file_length = os.path.getsize(file_path)
        podcast_xml = repo.get_contents(self.xml_path)
        xml_data = base64.b64decode(podcast_xml.content).decode('utf-8')
        xml_data = self.update_podcast_xml(xml_data, file_name, episode_title, episode_description, file_length)
        self.upload_to_github(self.xml_path, xml_data, f'Update podcast.xml with new episode: {file_name}', podcast_xml.sha)

    def upload_to_github(self, file_name, file_content, commit_message, sha=None):
        # Prepare API request headers
        token = os.environ.get("GH_KEY", None) or open("/Users/jong/.gh_token").read().strip()
        gh = Github(token)
        # Get the repository
        try:
            repo = gh.get_user().get_repo(self.repo)
        except:
            repo = gh.get_organization(self.org).get_repo(self.repo)

        if sha:
            repo.update_file(file_name, commit_message, file_content, sha)
        else:
            repo.create_file(file_name, commit_message, file_content)


# %%
class Episode:
    def __init__(self, episode_type='narration', podcast_args=("JonathanGrant", "jonathangrant.github.io", "podcasts/podcast.xml"), text_model=DEFAULT_MODEL, **chat_kwargs):
        """
        Kinds of episodes:
            pure narration - simple TTS
            simple podcast - Text to Podcast
            complex podcast?
        """
        self.episode_type = episode_type
        self.chat = PodcastChat(**chat_kwargs)
        self.chat_kwargs = chat_kwargs
        self.pod = PodcastRSSFeed(*podcast_args)
        self.text_model = text_model
        self.sounds = []
        self.texts = []

    def get_outline(self, n, topic=None):
        if topic is None: topic = self.chat._topic
        chat = Chat(f"""Write 
a concise plaintext outline with exactly {n} parts for a podcast titled {self.chat._podcast}.
Only return the parts and nothing else.
Do not include a conclusion or intro.
Do not write more than {n} parts.
Format it like this: 1. insert-title-here, 2. another-title-here, ...""".replace("\n", " "))
        resp = chat.message(model=self.text_model)
        chapter_pattern = re.compile(r'\d+\.\s+.*')
        chapters = chapter_pattern.findall(resp)
        if not chapters:
            logger.warning(f'Could not parse message for chapters! Message:\n{resp}')
        return chapters

    def step(self, msg=None, nparts=3):
        include = f" Remember to respond with the hosts names like {self.chat._hosts[0]}: and {self.chat._hosts[1]}:"
        msg = msg or self.chat._topic
        if self.episode_type == 'narration':
            outline = self.get_outline(msg, nparts)
            logger.info(f"Outline: {outline}")
            intro_txt, intro_aud = self.chat.step(f"Write the intro for a podcast about {msg}. The outline for the podcast is {', '.join(outline)}. Only write the introduction.{include}", model=self.text_model)
            self.sounds.append(intro_aud)
            self.texts.append(intro_txt)
            # Get parts
            for part in outline:
                logger.info(f"Part: {part}")
                part_txt, part_aud = self.chat.step(f"Write the next part: {part}.{include}", model=self.text_model)
                self.sounds.append(part_aud)
                self.texts.append(part_txt)
            # Get conclusion
            logger.info("Conclusion")
            part_txt, part_aud = self.chat.step(f"Write the conclusion. Remember, the outline was: {', '.join(outline)}.{include}", model=self.text_model)
            self.sounds.append(part_aud)
            self.texts.append(part_txt)
        elif self.episode_type == 'pure_tts':
            outline = None
            audio = self.chat.text2speech("\n".join([self.chat._hosts[i%2]+": "+x for i,x in enumerate(msg)]))
            self.sounds.append(audio)
            self.texts.extend(msg)
        return outline, '\n'.join(self.texts)

    def upload(self, title, descr):
        title_small = title.lower().replace(" ", "_")[:16] + str(uuid.uuid4())  # I had a filename too long once
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = os.path.join(tmpdir, "audio_file.mp3")
            with open(tmppath, "wb") as f:
                f.write(merge_mp3s(self.sounds))
            self.pod.upload_episode(tmppath, f"podcasts/audio/{title_small}.mp3", title, descr)


# %%
def merge_mp3s(mp3_bytes_list):
    """
    Merges multiple MP3 bytestrings into a single MP3 bytestring.
    
    :param mp3_bytes_list: List of MP3 bytestrings
    :return: Merged MP3 as bytestring
    """
    # Convert the first MP3 bytestring to an AudioSegment
    combined = pydub.AudioSegment.from_file(io.BytesIO(mp3_bytes_list[0]), format="mp3")

    # Loop through the rest of the MP3 bytestrings and append them
    for mp3_bytes in mp3_bytes_list[1:]:
        next_segment = pydub.AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        combined += next_segment

    # Export the combined audio to a bytestring
    combined_buffer = io.BytesIO()
    combined.export(combined_buffer, format="mp3")
    return combined_buffer.getvalue()


# %%
class AIChain:
    @classmethod
    def serial_message(cls, chat=None, msgs=None, systems=None):
        responses = []
        systems = systems or [None] * len(msgs)
        for msg, sys in tqdm(zip(msgs, systems), total=len(msgs), desc="Processing serial chat messages.", unit="message"):
            if sys is not None:
                chat._history[0]['content'] = sys
            responses.append(chat.message(next_msg=msg))
        return responses

    @classmethod
    def parallel_message(cls, chats=None, msgs=None, max_workers=8):
        assert len(chats) == len(msgs), "Lengths of chats and msgs must match."
        responses = [None] * len(chats)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as tpe:
            futures2idx = {tpe.submit(chat.message, next_msg=msg): i for i, chat, msg in zip(range(9**99), chats, msgs)}
            for future in tqdm(concurrent.futures.as_completed(futures2idx.keys()), total=len(futures2idx), desc="Processing parallel chats", unit="chat"):
                responses[futures2idx[future]] = future.result()
        return responses

# %%
# # %%time
# ep = Episode(
#     episode_type='narration',
#     topic="Hidden History: Unraveling 3 of History's Funniest Mysteries from the 1st Century",
#     max_length=29_000,
#     # text_model='gpt-4-1106-preview',
#     text_model='GOOGLE/gemini-pro',
# )
# outline, txt = ep.step(nparts='3')
# ep.upload("[Google Gemini] Hidden History: Unraveling 3 of History's Funniest Mysteries from the 1st Century", "Hidden History: Unraveling 3 of History's Funniest Mysteries from the 1st Century")

# %%

# %%

