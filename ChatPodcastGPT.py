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
import shutil
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
import os
import re
import io
import concurrent.futures
import tempfile
import shutil
import subprocess
import logging
import pydub # Still needed for fallback merge or potential segment validation
import IPython # For display in notebooks
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
except:
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
    """
    Generates speech from text using the OpenAI Text-to-Speech API.
    See: https://platform.openai.com/docs/guides/text-to-speech
    """

    VOICES = {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
    }

    MAN = "onyx"
    WOMAN = "coral"

    def __init__(self, voice='alloy', model='gpt-4o-mini-tts', api_key=None):
        """
        Initializes the OpenAI TTS client.

        Args:
            voice (str): The voice to use. Available voices:
                         alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer
                         Defaults to 'alloy'.
            model (str): The TTS model to use. Available models:
                         gpt-4o-mini-tts (recommended), tts-1, tts-1-hd
                         Defaults to 'gpt-4o-mini-tts'.
            api_key (str, optional): Your OpenAI API key. If None, the client
                                     will try to use the OPENAI_API_KEY environment
                                     variable. Defaults to None.
        """
        self.voice = voice
        self.model = model
        # Initialize the client once
        # If api_key is provided, use it; otherwise, OpenAI() will look for env var
        self.client = openai.OpenAI(api_key=api_key if api_key else openai.api_key) # Maintains compatibility if openai.api_key was set globally

    @RateLimited(95) # Keep your original decorator
    @jonlog.retry_with_logging() # Keep your original decorator
    def tts(self, text: str, instructions: str = None):
        """
        Converts text to speech.

        Args:
            text (str): The text content to synthesize.
            instructions (str, optional): Specific instructions for the speech
                                          generation (e.g., tone, accent, speed).
                                          Only applicable for models like gpt-4o-mini-tts.
                                          Defaults to None.

        Returns:
            bytes: The raw audio content (typically MP3 format by default).

        Raises:
            openai.APIError: If the API request fails.
        """
        params = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
        }
        # Only add instructions if provided and the model supports it (implicitly assuming gpt-4o-mini-tts does)
        if instructions:
             # Check if the model likely supports instructions
             # A simple check, you might want more robust logic if needed
            if 'gpt-4o' in self.model:
                params["instructions"] = instructions
            else:
                # Optionally log a warning if instructions are provided for non-compatible models
                print(f"Warning: 'instructions' parameter provided but model '{self.model}' might not support it.")
                # Do not pass instructions for models like tts-1/tts-1-hd


        response = self.client.audio.speech.create(**params)

        # The response object itself doesn't have .content directly in v1+ anymore for the standard call
        # You read the raw bytes from the response object itself if not streaming.
        # However, the previous code returned response.content. Let's check the actual response object structure.
        # According to docs and common usage for non-streaming, you get an httpx.Response.
        # Let's assume response.content works as before or adapt if needed.
        # If `response` is directly the httpx response object:
        return response.content # This should contain the raw audio bytes

        # If the structure changed significantly and response is NOT an httpx response,
        # you might need to adjust how you get the bytes. For example, if it returned
        # a custom object, you might need a different attribute or method.
        # But `response.content` is standard for non-streaming httpx responses.


    def tostring(self):
        return f"[OpenAITTS] voice='{self.voice}' model='{self.model}'"

    def __repr__(self):
        return self.tostring()

    def __str__(self):
        return self.tostring()


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
        possible += [OpenAITTS(voice=vid) for vid in OpenAITTS.VOICES]
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
        "claude-3.7-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
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
        "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-exp-03-25",
    }

    @classmethod
    def get_apikey(cls):
        return os.environ.get("GEMINI_API_KEY") or open(os.path.expanduser("~/.google_apikey")).read().strip()

    @classmethod
    def consolidate_messages(cls, message_list, keep_system=False):
        if not message_list:
            return []
        consolidated = []
        current_role = None
        current_content = ""

        for message in message_list:
            role = message.get("role")
            content = message.get("content", "")
    
            if role == "system" and not keep_system:
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

        # Consolidate messages
        consolidated_messages = cls.consolidate_messages(messages, keep_system=True)
        final_msg, system_msg = "Continue", None
        if consolidated_messages[-1]['role'] == 'user':
            final_msg = consolidated_messages.pop(-1)['content']
        if consolidated_messages[0]['role'] == 'system':
            system_msg = consolidated_messages.pop(0)['content']
        
        # Initialize the model
        model = google.generativeai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
            system_instruction=system_msg,
        )

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
        "claude-opus": "claude-3-opus-20240229",    # Large
        "claude-sonnet": "claude-3-sonnet-20240229", # Medium
        "claude-haiku": "claude-3-5-haiku-latest", # Small
        "claude-sonnet-3.5": "claude-3-5-sonnet-20240620", # Updated Sonnet
        "claude-sonnet-3.7": "claude-3-7-sonnet-20250219", # Latest Sonnet
    }
    
    @classmethod
    def get_apikey(cls):
        return os.environ.get("ANTHROPIC_APIKEY") or open('/Users/jong/.anthropic_apikey').read().strip()
    
    @classmethod
    def msg(cls, messages=None, model="claude-3-7-sonnet-20250219", **kwargs):
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
            messages=messages,
            system=system,
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
        return os.environ.get("TOGETHER_API_KEY") or open(os.path.expanduser("~/.together_apikey")).read().strip()

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
# DEFAULT_MODEL = 'gpt-4-1106-preview'
# DEFAULT_LENGTH  = 80_000
DEFAULT_MODEL = 'ANTHROPIC/claude-3-7-sonnet-20250219'
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
    def __init__(self, topic, podcast="award winning", max_length=DEFAULT_LENGTH,
                 hosts=['Tom', 'Jen'],
                 host_voices=[OpenAITTS(voice='verse'), OpenAITTS(voice='ash')], # Example voices
                 host_instructions=None, # <<< Added parameter
                 extra_system=None, system=None):
        """ (Docstring remains the same) """
        if system is None:
            system = f"""You are an {podcast} podcast with hosts {hosts[0]} and {hosts[1]}.
Respond with the hosts names before each line like {hosts[0]}: and {hosts[1]}:""".replace("\n", " ")
        if extra_system is not None:
            system = '\n'.join([system, extra_system])

        super().__init__(system, max_length=max_length)
        self._podcast = podcast
        self._topic = topic
        if len(hosts) != 2 or len(host_voices) != 2:
            raise ValueError("Requires exactly two hosts and two corresponding voices.")
        self._hosts = hosts
        # Updated history logic based on provided snippet
        if host_instructions:
             # Also include the main generation prompt along with instructions
             self._history.append({
                 "role": "user", "content": f"""Generate an informative, entertaining, and very detailed podcast episode about {topic}.
 Respond with the hosts names before each line like\n\n{hosts[0]}: ...\nand\n{hosts[1]}:...\n\nUse these specific instructions per host: {host_instructions}"""
             })
        else:
            # Original prompt if no instructions
            self._history.append({
                "role": "user", "content": f"""Generate an informative, entertaining, and very detailed podcast episode about {topic}.
 Respond with the hosts names before each line like\n\n{hosts[0]}: ...\nand\n{hosts[1]}:...\n"""
            })
        self._tts_h1, self._tts_h2 = host_voices
        self._host_instructions = host_instructions or {} # <<< Store instructions, default to empty dict

    def text2speech(self, text, spacing_ms=350): # spacing_ms still unused here
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as thread_pool:
            segment_index = 0 # Use simple counter for segment order
            submitted_futures = [] # Store futures in submission order

            # Inner function remains the same, accepting index 'i' for logging
            def write_audio(msg, i, voice, instructions=None, **kwargs):
                logger.info(f'Requesting TTS segment_index={i} for voice={voice}')
                can_accept_instructions = True

                tts_result = None
                if instructions and can_accept_instructions:
                    logger.info(f"  Passing instructions to segment_index={i}: '{instructions}'")
                    tts_result = voice.tts(msg, instructions=instructions)
                else:
                    if instructions and not can_accept_instructions:
                         logger.warning(f"  Instructions provided for segment_index={i} ('{instructions}') but {voice}.tts does not accept them. Calling without.")
                    tts_result = voice.tts(msg)
                logger.info(f'Received TTS segment_index={i} for voice={voice}')
                return tts_result # Return only the audio bytes

            # --- Text processing and job submission ---
            text = text.replace('\n', '!!!LINEBREAK!!!').replace('\\', '').replace('"', '')
            currline, currname = "", self._hosts[0] # Start with the first host
            name2voice = {self._hosts[0]: self._tts_h1, self._hosts[1]: self._tts_h2}

            host1_prefix = f"{self._hosts[0]}: "
            host2_prefix = f"{self._hosts[1]}: "

            for line in text.split("!!!LINEBREAK!!!"):
                line_strip = line.strip()
                if not line_strip: continue

                new_host_found = False
                potential_host = None
                line_content = ""

                if line.startswith(host1_prefix):
                    potential_host = self._hosts[0]
                    line_content = line[len(host1_prefix):]
                    new_host_found = True
                elif line.startswith(host2_prefix):
                    potential_host = self._hosts[1]
                    line_content = line[len(host2_prefix):]
                    new_host_found = True
                else:
                    line_content = line # Treat as continuation

                if new_host_found:
                    if currline: # Process the previous host's accumulated lines
                        text_to_speak = currline.strip()
                        if text_to_speak: # Avoid submitting empty strings
                             instruction = self._host_instructions.get(currname, None)
                             logger.info(f"Submitting segment {segment_index} for {currname}: '{text_to_speak[:50]}...'")
                             future = thread_pool.submit(
                                 write_audio,
                                 text_to_speak,
                                 segment_index, # Pass the current segment index
                                 name2voice[currname],
                                 instructions=instruction
                             )
                             submitted_futures.append(future) # Store future in order
                             segment_index += 1
                    currline = line_content # Start new line content
                    currname = potential_host # Set the new current host
                else:
                    currline += " " + line_content # Accumulate content

            # Process the last accumulated line
            if currline:
                 text_to_speak = currline.strip()
                 if text_to_speak: # Avoid submitting empty strings for the last segment
                     instruction = self._host_instructions.get(currname, None)
                     logger.info(f"Submitting segment {segment_index} for {currname}: '{text_to_speak[:50]}...'")
                     future = thread_pool.submit(
                         write_audio,
                         text_to_speak,
                         segment_index, # Pass the current segment index
                         name2voice[currname],
                         instructions=instruction
                     )
                     submitted_futures.append(future) # Store final future in order
                     segment_index += 1

            # --- Collect results in submission order ---
            logger.info(f"Waiting for {len(submitted_futures)} audio segments to complete...")
            audios = []
            for i, future in enumerate(submitted_futures):
                 try:
                     # .result() will wait for the future to complete
                     result = future.result()
                     audios.append(result)
                     logger.info(f"Collected result for segment {i}")
                 except Exception as e:
                     logger.error(f"Error getting result for segment {i}: {e}")
                     # Decide how to handle errors: skip segment, raise error, etc.
                     # For now, let's append None or empty bytes to avoid breaking merge_mp3s if possible
                     audios.append(b"") # Append empty bytes on error

            # --- Concatenate files ---
            logger.info('Concatenating audio segments in order...')
            audio = merge_mp3s(audios) # Pass the ordered list
            logger.info('Done with audio synthesis and merging!')
            IPython.display.display(IPython.display.Audio(audio, autoplay=False))
            return audio

    def step(self, msg=None, skip_aud=False, ret_aud=True, min_length=None, **kwargs):
        # (Step method remains the same)
        text_response = self.message(msg, **kwargs)
        logger.info(f"Generated text response:\n{text_response}")
        if min_length is not None and len(text_response) < min_length:
            raise ValueError(f"Message [{text_response[:100]}...] is shorter than {min_length=}")
        if skip_aud:
            return text_response
        audio_response = self.text2speech(text_response)
        if ret_aud:
            return text_response, audio_response
        return text_response


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
    def __init__(self, episode_type='narration', podcast_args=("JonathanGrant", "jonathangrant.github.io", "podcasts/podcast.xml"), text_model=DEFAULT_MODEL, desc="hilarious", **chat_kwargs):
        """
        Kinds of episodes:
            pure narration - simple TTS
            simple podcast - Text to Podcast
            complex podcast?
        """
        self.episode_type = episode_type
        self.chat = PodcastChat(**chat_kwargs)
        self.chat_kwargs = chat_kwargs
        # self.pod = PodcastRSSFeed(*podcast_args)
        self.text_model = text_model
        self.sounds = []
        self.texts = []
        self._desc = desc

    def get_outline(self, n, topic=None):
        if topic is None: topic = self.chat._topic
        chat = Chat(f"""Write 
a concise plaintext outline with exactly {n} parts for an epsiode of the {self._desc} podcast titled {self.chat._podcast}.
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
class EpisodeJoe(Episode):
    def get_outline(self, n, topic=None):
        if topic is None: topic = self.chat._topic
        chat = Chat(f"""Write 
    a concise plaintext outline with exactly {n} parts for an episode of the {self._desc} podcast titled {self.chat._podcast}.
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
        msg = msg or self.chat._topic
        hosts_format = f"{self.chat._hosts[0]} and {self.chat._hosts[1]}"
        
        if self.episode_type == 'narration':
            outline = self.get_outline(nparts, msg)
            logger.info(f"Outline: {outline}")
            
            # Generate intro script
            prompt = f"""Write the {self._desc} ACTUAL DIALOGUE script for the introduction of a podcast episode about {msg}.
    The hosts are {hosts_format}.
    Write ONLY the back-and-forth conversation between the hosts.
    Format each line with the speaker's name followed by a colon, like this:
    {self.chat._hosts[0]}: [what they say]
    {self.chat._hosts[1]}: [what they say]"""
            
            intro_txt, intro_aud = self.chat.step(prompt, model=self.text_model)
            self.sounds.append(intro_aud)
            self.texts.append(intro_txt)
            
            # Get parts
            for part in outline:
                logger.info(f"Part: {part}")
                part_prompt = f"""Write the {self._desc} ACTUAL DIALOGUE script for this segment of the podcast: {part}
    The hosts are {hosts_format}.
    Write ONLY the back-and-forth conversation as it would happen in real time.
    Format each line with the speaker's name followed by a colon, like this:
    {self.chat._hosts[0]}: [what they say]
    {self.chat._hosts[1]}: [what they say]"""
                
                part_txt, part_aud = self.chat.step(part_prompt, model=self.text_model)
                self.sounds.append(part_aud)
                self.texts.append(part_txt)
            
            # Get conclusion
            conclusion_prompt = f"""Write the {self._desc} ACTUAL DIALOGUE script for the conclusion of this podcast.
    The hosts are {hosts_format}.
    Write ONLY the back-and-forth closing conversation.
    Format each line with the speaker's name followed by a colon, like this:
    {self.chat._hosts[0]}: [what they say]
    {self.chat._hosts[1]}: [what they say]"""
            
            part_txt, part_aud = self.chat.step(conclusion_prompt, model=self.text_model)
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
        tmppath = os.path.join('/Users/jong/Documents/', f"{title_small}.mp3")
        with open(tmppath, "wb") as f:
            f.write(merge_mp3s(self.sounds))


# %%
def merge_mp3s_pydub(mp3_bytes_list):
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

def merge_mp3s(mp3_bytes_list, use_ffmpeg=True):
    """
    Merges multiple MP3 bytestrings into a single MP3 bytestring,
    preferring FFmpeg for efficiency if available and requested.

    FFmpeg uses stream copy (-c copy), which is fast and low-resource,
    but requires input MP3s to have compatible parameters (sample rate, channels).
    If FFmpeg fails or is not used, it falls back to the pydub method.

    :param mp3_bytes_list: List of MP3 bytestrings.
    :param use_ffmpeg: Boolean, whether to attempt using FFmpeg first.
    :return: Merged MP3 as bytestring, or None on error.
    """
    if not mp3_bytes_list:
        logger.warning("Received empty list for MP3 merging.")
        return b""

    # Filter out potentially invalid/empty segments early
    valid_mp3_bytes = [bs for bs in mp3_bytes_list if isinstance(bs, bytes) and len(bs) > 100] # Basic size check
    if not valid_mp3_bytes:
        logger.warning("No valid MP3 byte strings found after filtering.")
        return b""

    # --- Attempt FFmpeg Method ---
    ffmpeg_path = shutil.which('ffmpeg') if use_ffmpeg else None
    if ffmpeg_path:
        logger.info(f"Attempting efficient merge using FFmpeg ({ffmpeg_path}) for {len(valid_mp3_bytes)} segments...")
        temp_files = []
        list_file_path = None
        output_file_path = None

        try:
            # Create temporary files for each MP3 segment
            logger.debug("Creating temporary files for input segments...")
            for i, mp3_bytes in enumerate(valid_mp3_bytes):
                # Suffix helps ffmpeg identify format sometimes, though not strictly necessary here
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_file.write(mp3_bytes)
                temp_files.append(temp_file.name)
                temp_file.close() # Close the file handle

            if not temp_files:
                 raise ValueError("Failed to create any temporary input files.")

            # Create a temporary file list for FFmpeg's concat demuxer
            logger.debug("Creating temporary file list...")
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt") as list_file:
                list_file_path = list_file.name
                for temp_fpath in temp_files:
                    # FFmpeg requires 'file' keyword and proper quoting/escaping if paths have spaces/special chars
                    # Using basic paths from NamedTemporaryFile should be safe, but use os.path.abspath if needed
                    # Use forward slashes even on Windows for FFmpeg concat demuxer list file
                    safe_path = temp_fpath.replace('\\', '/')
                    list_file.write(f"file '{safe_path}'\n")

            # Create a temporary output file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_out_file:
                 output_file_path = temp_out_file.name

            # Construct and run the FFmpeg command
            # -f concat: Use the concat demuxer
            # -safe 0: Necessary for using potentially complex/absolute paths in the list file
            # -i list.txt: Input file list
            # -c copy: Copy streams without re-encoding (FAST, LOW RESOURCE)
            # -y: Overwrite output file without asking
            command = [
                ffmpeg_path,
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file_path,
                '-c', 'copy',
                output_file_path
            ]
            logger.info(f"Running FFmpeg command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False) # Use check=False to handle errors manually

            # Check FFmpeg execution result
            if result.returncode != 0:
                logger.error(f"FFmpeg failed with return code {result.returncode}.")
                logger.error(f"FFmpeg stderr:\n{result.stderr}")
                # Fallback or raise error - let's fallback for now
                raise RuntimeError("FFmpeg concatenation failed.")
            else:
                logger.info("FFmpeg merge successful.")
                # Read the merged content from the temporary output file
                with open(output_file_path, 'rb') as f_out:
                    merged_bytes = f_out.read()
                return merged_bytes

        except Exception as ffmpeg_err:
            logger.warning(f"FFmpeg merge failed: {ffmpeg_err}. Trying fallback to pydub method.")
            # Explicitly trigger fallback if FFmpeg fails midway
            return merge_mp3s_pydub(valid_mp3_bytes) # Pass the filtered list

        finally:
            # Clean up temporary files
            logger.debug("Cleaning up temporary files...")
            if list_file_path and os.path.exists(list_file_path):
                os.remove(list_file_path)
            if output_file_path and os.path.exists(output_file_path):
                os.remove(output_file_path)
            for temp_f in temp_files:
                if os.path.exists(temp_f):
                    os.remove(temp_f)
            logger.debug("Temporary file cleanup finished.")

    else:
        # --- Fallback to pydub Method ---
        if use_ffmpeg:
             logger.warning("FFmpeg not found or use_ffmpeg=False. Falling back to pydub merge (less efficient).")
        else:
             logger.info("Using pydub merge method as requested.")

        return merge_mp3s_pydub(valid_mp3_bytes) # Pass the filtered list



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
# import os
# import re
# import io
# import concurrent.futures
# import tempfile
# import shutil
# import subprocess
# import logging
# import pydub # Still needed for fallback merge or potential segment validation
# import IPython # For display in notebooks
# joe = """Delivery: Methodical and rhythmic, with a steady conversational pace punctuated by moments of intense focus, spontaneous digressions, and deep, rumbling laughter that occasionally builds into breathless, high-pitched cackling.
# Voice: Husky and resonant, carrying an approachable authority that becomes notably animated when discussing topics of personal interest, from psychedelic experiences to evolutionary biology or controversial viewpoints.
# Tone: Persistently curious and democratically skeptical, blending street-smart pragmatism with genuine wonder at complex ideas. Effortlessly shifts between casual bro-talk, earnest philosophical inquiry, and wide-eyed astonishment.
# Pronunciation: Direct and unadorned with subtle Massachusetts undertones, often stretching key syllables for emphasis and impact. Frequently deploys characteristic expressions like "That's wild," "A hundred percent," and "Look into it" with distinctive cadence and conviction."""
# joe2 = """Delivery: Relaxed yet engaging, with natural conversational flow punctuated by thoughtful silences, sudden bursts of enthusiasm, and authentic, contagious laughter that often evolves into wheezing fits.
# Voice: Gravelly and masculine, maintaining a casual confidence that intensifies when tackling subjects he's passionate about, from combat sports to psychedelics or fringe theories.
# Tone: Inquisitive and unpretentious, combining everyman relatability with genuine intellectual curiosity. Seamlessly transitions between lighthearted humor, profound questions, and incredulous reactions.
# Pronunciation: Straightforward with subtle regional inflections, frequently emphasizing certain words with elongated vowels. Regularly employs signature phrases like "That's fascinating," "Have you ever tried..." and "Jamie, pull that up" with distinctive rhythm and emphasis."""

# hosts = ['Joe', 'Joh']

# inst = {hosts[0]: joe, hosts[1]: joe2}
# ep = EpisodeJoe(
#     episode_type='narration',
#     topic="Considering todays Logitech business, what kind of products will Logitech bring to market in three years when AI is dominating everything we do as humans, both work an play",
#     max_length=200_000,
#     podcast="The Joe Rogan Experience",
#     text_model=DEFAULT_MODEL,
#     host_instructions=inst,
#     hosts=hosts,
#     desc=""
# )
# outline, txt = ep.step(nparts='1')
# ep.upload("[Claude 4o] logihio", "Test7")

# %%
# import os
# import re
# import io
# import concurrent.futures
# import tempfile
# import shutil
# import subprocess
# import logging
# import pydub # Still needed for fallback merge or potential segment validation
# import IPython # For display in notebooks

# # Assume logger, OpenAITTS, merge_mp3s, merge_mp3s_pydub are defined above this point
# # Example logger setup if needed:
# try:
#     logger
# except NameError:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)


# class SeinfeldAudioGenerator:
#     """
#     Generates audio for a Seinfeld script, handling character voices,
#     inline instructions, TTS generation, merging, and saving.
#     Stores intermediate results as instance attributes.
#     """
#     # Default character voice definitions (can be overridden in __init__)
#     DEFAULT_CHARACTER_VOICES = {
#         "JERRY": {
#             "voice_name": "ash",
#             "instructions": "Speak in a slightly high-pitched, observational, and often questioning tone, typical of a stand-up comedian. Emphasize ironic points with subtle sarcasm. Pace is generally moderate but can speed up during rants."
#         },
#         "GEORGE": {
#             "voice_name": "ballad",
#             "instructions": "Sound like a neurotic, often whining or complaining character. Use a slightly nasal quality. Delivery can range from low and scheming to loud and panicky outbursts. Frequently uses upward inflections at the end of sentences, even when not asking a question."
#         },
#         "ELAINE": {
#             "voice_name": "sage",
#             "instructions": "Speak with expressive, sometimes forceful energy. Tone can be sarcastic, exasperated, or enthusiastic. Use clear pronunciation but allow for moments of loud frustration or laughter. Pace is often quick and assertive."
#         },
#         "KRAMER": {
#             "voice_name": "verse",
#             "instructions": "Use an eccentric, physically animated voice. Incorporate sudden bursts of energy, awkward pauses, clicks, and unique vocalizations. Delivery is unpredictable, often jerky, with wide variations in pitch and volume. Sound slightly manic and unconventional."
#         },
#         "NEWMAN": {
#             "voice_name": "coral",
#             "instructions": "Manly voice. Speak with a sneering, often antagonistic tone, especially towards Jerry. Delivery should sound somewhat dramatic and self-important, with clear enunciation. Add a hint of passive-aggression."
#         },
#         "DEFAULT": {
#             "voice_name": "alloy",
#             "instructions": "Speak in a neutral, standard American accent."
#         }
#     }

#     def __init__(self, script_filepath, output_dir=None, tts_model='gpt-4o-mini-tts', max_workers=5, character_voices=None):
#         """
#         Initializes the audio generator.

#         Args:
#             script_filepath (str): Path to the input text script file.
#             output_dir (str, optional): Directory to save the output MP3 and debug segments.
#                                         If None, uses the script's directory. Defaults to None.
#             tts_model (str, optional): The OpenAI TTS model to use. Defaults to 'gpt-4o-mini-tts'.
#             max_workers (int, optional): Max threads for concurrent TTS calls. Defaults to 5.
#             character_voices (dict, optional): Override default character voice definitions.
#                                                 Defaults to None (uses DEFAULT_CHARACTER_VOICES).
#         """
#         self.script_filepath = script_filepath
#         self.output_dir = output_dir
#         self.tts_model = tts_model
#         self.max_workers = max_workers
#         self.character_voices = character_voices or self.DEFAULT_CHARACTER_VOICES.copy() # Use default if none provided

#         # --- State Variables ---
#         self.script_content_lines = []
#         self.dialogue_segments = [] # List of tuples: (index, char_name, text_for_tts, tts_instance_key, combined_instructions)
#         self.tts_instances = {} # Dict storing initialized TTS clients {voice_name: instance}
#         self.raw_audio_results = {} # Dict storing successful results {index: audio_bytes}
#         self.ordered_audio_bytes = [] # List of successful audio bytes in order
#         self.merged_audio = None # Bytes of the final merged audio
#         self.final_output_path = None # Path where the final audio is saved
#         self.debug_segments_path = None # Path where debug segments were saved
#         self.errors = [] # List to store errors encountered during processing

#         # --- Regex (defined once) ---
#         self.dialogue_pattern = re.compile(r'^\s*([A-Z][A-Z\s]+):\s*(.*)')
#         self.inline_instruction_pattern = re.compile(r'\*\((.*?)\)\*')
#         self.inline_removal_pattern = re.compile(r'\s*\*\([^)]*\)\*\s*')

#         # --- Determine Base Save Directory ---
#         self.script_dir = os.path.dirname(self.script_filepath)
#         self.base_save_directory = self.output_dir if self.output_dir else self.script_dir
#         self.script_basename = os.path.splitext(os.path.basename(self.script_filepath))[0]


#     def _log_error(self, message, exception=None):
#         """Helper to log errors and store them."""
#         full_message = f"{message}{f': {exception}' if exception else ''}"
#         logger.error(full_message, exc_info=exception is not None)
#         self.errors.append(full_message)

#     def _initialize_tts(self):
#         """Initializes TTS instances for all required voices."""
#         logger.info("Initializing TTS instances...")
#         required_voices = {details['voice_name'] for details in self.character_voices.values()}

#         for voice_name in required_voices:
#             if voice_name not in self.tts_instances:
#                 try:
#                     # Assumes OpenAITTS class is globally available
#                     self.tts_instances[voice_name] = OpenAITTS(voice=voice_name, model=self.tts_model)
#                     logger.info(f"Initialized TTS for voice '{voice_name}': {self.tts_instances[voice_name]}")
#                 except NameError:
#                     self._log_error("OpenAITTS class definition not found. Cannot initialize TTS.")
#                     return False
#                 except Exception as e:
#                     self._log_error(f"Error initializing TTS instance for '{voice_name}'", e)
#                     return False
#         return True

#     def parse_script(self):
#         """Reads the script file and parses dialogue segments."""
#         logger.info(f"Parsing script file: {self.script_filepath}")
#         if not os.path.exists(self.script_filepath):
#             self._log_error(f"Script file not found: {self.script_filepath}")
#             return False

#         try:
#             with open(self.script_filepath, 'r', encoding='utf-8') as f:
#                 self.script_content_lines = f.readlines()
#         except Exception as e:
#             self._log_error(f"Error reading script file {self.script_filepath}", e)
#             return False

#         self.dialogue_segments = [] # Reset if called multiple times
#         segment_index = 0
#         for line_num, line in enumerate(self.script_content_lines, 1):
#             match = self.dialogue_pattern.match(line)
#             if match:
#                 character_name = match.group(1).strip()
#                 dialogue_text = match.group(2).strip()

#                 if not dialogue_text: continue # Skip empty

#                 # Find voice config
#                 char_config = self.character_voices.get(character_name, self.character_voices["DEFAULT"])
#                 if character_name not in self.character_voices:
#                      logger.warning(f"Line {line_num}: Character '{character_name}' not explicitly defined. Using default voice.")

#                 base_instructions = char_config["instructions"]
#                 tts_voice_name = char_config["voice_name"] # Key to look up instance
#                 final_instructions = base_instructions
#                 text_for_tts = dialogue_text

#                 # Check for inline instructions
#                 inline_match = self.inline_instruction_pattern.search(dialogue_text)
#                 if inline_match:
#                     inline_instruction = inline_match.group(1).strip()
#                     # You might want to adjust how instructions are combined:
#                     final_instructions = f"Follow this specific instruction: '{inline_instruction}'. Also adhere to the general character guidance: {base_instructions}"
#                     text_for_tts = self.inline_removal_pattern.sub(' ', dialogue_text).strip()
#                     logger.info(f"Line {line_num}: Found inline instruction '{inline_instruction}' for {character_name}. Cleaned text: '{text_for_tts[:50]}...'")

#                 if not text_for_tts:
#                     logger.warning(f"Line {line_num}: Dialogue text became empty after removing instruction for {character_name}. Skipping segment.")
#                     continue

#                 self.dialogue_segments.append(
#                     (segment_index, character_name, text_for_tts, tts_voice_name, final_instructions)
#                 )
#                 segment_index += 1

#         if not self.dialogue_segments:
#             self._log_error("No valid dialogue segments found or parsed in the script.")
#             return False

#         logger.info(f"Parsed {len(self.dialogue_segments)} dialogue segments.")
#         return True

#     def _generate_speech_segment_task(self, index, char_name, text, tts_instance_key, instructions):
#         """Task function for generating single audio segment (used by thread pool)."""
#         try:
#             tts_instance = self.tts_instances.get(tts_instance_key)
#             if not tts_instance:
#                  raise ValueError(f"TTS instance for key '{tts_instance_key}' not found.")

#             logger.info(f"Requesting TTS for segment {index} ({char_name}): '{text[:50]}...' using {tts_instance}")

#             # Pass instructions based on instance type and model
#             audio_bytes = None
#             if isinstance(tts_instance, OpenAITTS) and 'gpt-4o' in tts_instance.model:
#                 # logger.debug(f"Segment {index} instructions: {instructions}")
#                 audio_bytes = tts_instance.tts(text, instructions=instructions)
#                 logger.info(f"Received TTS for segment {index} ({char_name}) with instructions.")
#             else:
#                 if instructions and isinstance(tts_instance, OpenAITTS):
#                     logger.warning(f"Segment {index} ({char_name}): Instructions provided but model {tts_instance.model} might not support them. Calling TTS without.")
#                 elif instructions and not isinstance(tts_instance, OpenAITTS):
#                     logger.warning(f"Segment {index} ({char_name}): TTS instance is not OpenAITTS, instructions ignored.")
#                 audio_bytes = tts_instance.tts(text)
#                 logger.info(f"Received TTS for segment {index} ({char_name}) without instructions.")

#             # Validation
#             if not audio_bytes or len(audio_bytes) < 100: # Check for empty or too small
#                 logger.warning(f"TTS returned invalid/empty audio for segment {index} ({char_name}). Size={len(audio_bytes) if audio_bytes else 0}. Skipping.")
#                 return index, None
#             return index, audio_bytes

#         except Exception as e:
#             self._log_error(f"Error in TTS task for segment {index} ({char_name})", e)
#             return index, None # Return index and None on error

#     def generate_segment_audio(self):
#         """Generates audio for all parsed dialogue segments concurrently."""
#         if not self.dialogue_segments:
#             self._log_error("Cannot generate audio, no dialogue segments parsed.")
#             return False
#         if not self.tts_instances:
#              self._log_error("Cannot generate audio, TTS instances not initialized.")
#              return False

#         logger.info(f"Generating audio for {len(self.dialogue_segments)} segments using up to {self.max_workers} workers...")
#         self.raw_audio_results = {} # Reset results
#         futures_to_index = {}

#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             for index, char_name, text_to_speak, tts_instance_key, combined_instructions in self.dialogue_segments:
#                 future = executor.submit(
#                     self._generate_speech_segment_task,
#                     index, char_name, text_to_speak, tts_instance_key, combined_instructions
#                 )
#                 futures_to_index[future] = index

#             for future in concurrent.futures.as_completed(futures_to_index):
#                 original_index = futures_to_index[future]
#                 try:
#                     idx, audio_data = future.result()
#                     if audio_data:
#                         self.raw_audio_results[idx] = audio_data
#                     # else: Error/Warning already logged in _generate_speech_segment_task
#                 except Exception as e:
#                      # This catches errors during future.result() itself, unlikely if task handles errors
#                     self._log_error(f"Error retrieving result for segment {original_index}", e)

#         if not self.raw_audio_results:
#             logger.warning("No audio segments were successfully generated.")
#             # Don't necessarily return False, allow inspection of errors
#             # Maybe return False if *zero* segments succeeded?
#             return len(self.raw_audio_results) > 0

#         logger.info(f"Successfully generated audio for {len(self.raw_audio_results)} out of {len(self.dialogue_segments)} segments.")
#         return True

#     def save_debug_segments(self):
#         """Saves successfully generated individual audio segments to disk."""
#         if not self.raw_audio_results:
#             logger.info("No raw audio results to save as debug segments.")
#             return False

#         self.debug_segments_path = os.path.join(self.base_save_directory, "_debug_audio_segments")
#         logger.info(f"Saving {len(self.raw_audio_results)} individual audio segments for debugging to: {self.debug_segments_path}")

#         try:
#             os.makedirs(self.debug_segments_path, exist_ok=True)

#             # Correctly iterate over the dictionary items
#             for index, audio_bytes in self.raw_audio_results.items():
#                 if not audio_bytes: continue # Should not happen if stored, but check

#                 try:
#                     # Retrieve character name for filename
#                     # dialogue_segments: (idx, char_name, text, instance_key, instruction)
#                     character_name = self.dialogue_segments[index][1]
#                     safe_char_name = character_name.replace(' ', '_').strip()
#                 except (IndexError, TypeError) as e: # Catch potential errors accessing dialogue_segments
#                     self._log_error(f"Error getting character name for debug segment index {index}", e)
#                     safe_char_name = "UnknownChar"

#                 segment_filename = f"segment_{index:03d}_{safe_char_name}.mp3"
#                 segment_filepath = os.path.join(self.debug_segments_path, segment_filename)

#                 try:
#                     with open(segment_filepath, 'wb') as seg_file:
#                         seg_file.write(audio_bytes)
#                 except IOError as e:
#                     self._log_error(f"Failed to save debug segment {segment_filepath}", e)

#             return True # Indicate that saving was attempted

#         except Exception as e:
#             self._log_error(f"Could not create or write to debug directory {self.debug_segments_path}", e)
#             return False


#     def merge_audio(self, use_ffmpeg=True):
#         """Merges the generated audio segments in order."""
#         if not self.raw_audio_results:
#             self._log_error("No audio segments available to merge.")
#             return False

#         logger.info("Preparing to merge audio segments...")
#         self.ordered_audio_bytes = []
#         missing_count = 0
#         for i in range(len(self.dialogue_segments)):
#             audio_segment = self.raw_audio_results.get(i)
#             if audio_segment:
#                 self.ordered_audio_bytes.append(audio_segment)
#             else:
#                 missing_count += 1
#                 # logger.warning(f"Audio segment for original index {i} was missing or invalid, skipping in merge.")

#         if missing_count > 0:
#              logger.warning(f"{missing_count} audio segments were missing or invalid and will be skipped in the final merge.")

#         if not self.ordered_audio_bytes:
#             self._log_error("No valid audio segments remained after ordering. Cannot merge.")
#             return False

#         logger.info(f"Merging {len(self.ordered_audio_bytes)} audio segments in order...")
#         try:
#             # Assumes merge_mp3s function is globally available
#             self.merged_audio = merge_mp3s(self.ordered_audio_bytes, use_ffmpeg=use_ffmpeg)
#             if self.merged_audio:
#                  logger.info("Audio segments merged successfully.")
#                  return True
#             else:
#                  # merge_mp3s should log its own errors, but we add one here too
#                  self._log_error("Merging process returned no data.")
#                  return False
#         except NameError:
#             self._log_error("merge_mp3s function definition not found. Cannot merge audio.")
#             return False
#         except Exception as e:
#             self._log_error("Error during merging process", e)
#             return False

#     def save_final_audio(self):
#         """Saves the merged audio to the final output file."""
#         if not self.merged_audio:
#             self._log_error("No merged audio data available to save.")
#             return None # Return None for path if not saved

#         output_filename = f"{self.script_basename}_audio.mp3"
#         self.final_output_path = os.path.join(self.base_save_directory, output_filename)

#         logger.info(f"Saving final merged audio to: {self.final_output_path}")
#         try:
#             os.makedirs(self.base_save_directory, exist_ok=True)
#             with open(self.final_output_path, 'wb') as f_out:
#                 f_out.write(self.merged_audio)
#             logger.info("Final audio saved successfully.")

#             # Optional: Display audio in Jupyter environment
#             self._display_audio_if_possible(self.final_output_path)

#             return self.final_output_path # Return the path on success
#         except Exception as e:
#             self._log_error(f"Error saving merged audio file to {self.final_output_path}", e)
#             self.final_output_path = None # Reset path on error
#             return None

#     def _display_audio_if_possible(self, audio_path):
#         """Try displaying audio in IPython if available."""
#         try:
#             if 'IPython' in globals() and IPython.get_ipython():
#                  IPython.display.display(IPython.display.Audio(audio_path))
#         except Exception as e:
#              logger.debug(f"Could not display audio in IPython: {e}")
#              pass # Ignore if IPython display fails


#     def process(self, save_debug=True, merge_with_ffmpeg=True):
#         """
#         Runs the full audio generation pipeline:
#         Initialize TTS -> Parse Script -> Generate Audio -> [Save Debug Segments] -> Merge Audio -> Save Final Audio

#         Args:
#             save_debug (bool): If True, save individual segments after generation.
#             merge_with_ffmpeg (bool): If True, attempt to use FFmpeg for merging.

#         Returns:
#             str or None: The path to the final saved audio file if successful, otherwise None.
#         """
#         self.errors = [] # Clear previous errors if re-processing

#         if not self._initialize_tts():
#             return None # Error already logged

#         if not self.parse_script():
#             return None # Error already logged

#         if not self.generate_segment_audio() and not self.raw_audio_results:
#              # If generation returns false AND there are zero results, stop.
#              logger.error("Audio generation failed completely.")
#              return None

#         if save_debug:
#             self.save_debug_segments() # Attempt saving, don't stop pipeline if only this fails

#         if not self.merge_audio(use_ffmpeg=merge_with_ffmpeg):
#             # Merging failed, but self.raw_audio_results might still be useful
#             logger.error("Audio merging failed. Check errors and debug segments if saved.")
#             return None # Stop before saving final file

#         final_path = self.save_final_audio()
#         # final_path will be None if saving failed

#         if not final_path and self.errors:
#              logger.error("Processing finished with errors. Final audio not saved.")
#         elif not final_path:
#              logger.error("Processing finished, but final audio could not be saved for unknown reasons.")
#         else:
#              logger.info("Processing finished successfully.")

#         return final_path

# %%
# script_file = '/Users/jong/Documents/PodcastGPT/tv_show_workspaces/20250330_191704/Seinfeld_hilarious_Seinfeld_episode_abo/final_script_3acts.txt'

# # Instantiate the generator
# generator = SeinfeldAudioGenerator(script_file) # , output_dir='/custom/output/path' if needed)

# # Run the process
# final_audio_path = generator.process(save_debug=True, merge_with_ffmpeg=True)

# if final_audio_path:
#     print(f"Success! Final audio saved to: {final_audio_path}")
# else:
#     print("Processing failed or did not complete.")
#     print("Check generator state for details:")
#     print(f"  - Parsed Segments: {len(generator.dialogue_segments)}")
#     print(f"  - Generated Audio Segments: {len(generator.raw_audio_results)}")
#     if generator.debug_segments_path:
#         print(f"  - Debug Segments saved to: {generator.debug_segments_path}")
#     if generator.ordered_audio_bytes and not generator.merged_audio:
#         print(f"  - Merging failed, but {len(generator.ordered_audio_bytes)} ordered segments exist.")
#     print(f"  - Errors encountered: {len(generator.errors)}")
#     for i, err in enumerate(generator.errors):
#         print(f"    Error {i+1}: {err}")


# %%
# import os
# import re
# import io
# import concurrent.futures
# import tempfile
# import shutil
# import subprocess
# import logging
# import pydub # Still needed for fallback merge or potential segment validation
# import IPython # For display in notebooks

# # Assume logger, OpenAITTS, merge_mp3s, merge_mp3s_pydub are defined above this point
# # Example logger setup if needed:
# try:
#     logger
# except NameError:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)


# class SeinfeldAudioGenerator:
#     """
#     Generates audio for a Seinfeld script, handling character voices,
#     inline instructions (**adjective phrase**), TTS generation, merging, and saving.
#     Stores intermediate results as instance attributes.
#     """
#     # Default character voice definitions (can be overridden in __init__)
#     DEFAULT_CHARACTER_VOICES = {
#         "JERRY": {
#             "voice_name": "ash",
#             "instructions": "Speak in a slightly high-pitched, observational, and often questioning tone, typical of a stand-up comedian. Emphasize ironic points with subtle sarcasm. Pace is generally moderate but can speed up during rants."
#         },
#         "GEORGE": {
#             "voice_name": "ballad",
#             "instructions": "Sound like a neurotic, often whining or complaining character. Use a slightly nasal quality. Delivery can range from low and scheming to loud and panicky outbursts. Frequently uses upward inflections at the end of sentences, even when not asking a question."
#         },
#         "ELAINE": {
#             "voice_name": "sage",
#             "instructions": "Speak with expressive, sometimes forceful energy. Tone can be sarcastic, exasperated, or enthusiastic. Use clear pronunciation but allow for moments of loud frustration or laughter. Pace is often quick and assertive."
#         },
#         "KRAMER": {
#             "voice_name": "verse",
#             "instructions": "Use an eccentric, physically animated voice. Incorporate sudden bursts of energy, awkward pauses, clicks, and unique vocalizations. Delivery is unpredictable, often jerky, with wide variations in pitch and volume. Sound slightly manic and unconventional."
#         },
#         "NEWMAN": {
#             "voice_name": "coral",
#             "instructions": "Manly voice. Speak with a sneering, often antagonistic tone, especially towards Jerry. Delivery should sound somewhat dramatic and self-important, with clear enunciation. Add a hint of passive-aggression."
#         },
#         "DEFAULT": {
#             "voice_name": "alloy",
#             "instructions": "Speak in a neutral, standard American accent."
#         }
#     }

#     def __init__(self, script_filepath, output_dir=None, tts_model='gpt-4o-mini-tts', max_workers=5, character_voices=None):
#         """
#         Initializes the audio generator.

#         Args:
#             script_filepath (str): Path to the input text script file. Expected format:
#                                     CHARACTER: **Optional instruction** Dialogue text
#             output_dir (str, optional): Directory to save the output MP3 and debug segments.
#             tts_model (str, optional): The OpenAI TTS model to use.
#             max_workers (int, optional): Max threads for concurrent TTS calls.
#             character_voices (dict, optional): Override default character voice definitions.
#         """
#         self.script_filepath = script_filepath
#         self.output_dir = output_dir
#         self.tts_model = tts_model
#         self.max_workers = max_workers
#         self.character_voices = character_voices or self.DEFAULT_CHARACTER_VOICES.copy()

#         # --- State Variables ---
#         self.script_content_lines = []
#         self.dialogue_segments = []
#         self.tts_instances = {}
#         self.raw_audio_results = {}
#         self.ordered_audio_bytes = []
#         self.merged_audio = None
#         self.final_output_path = None
#         self.debug_segments_path = None
#         self.errors = []

#         # --- Regex (Updated for **adjective phrase**) ---
#         self.dialogue_pattern = re.compile(r'^\s*([A-Z][A-Z\s]+):\s*(.*)')
#         # Matches **anything non-greedy between double asterisks**
#         self.inline_instruction_pattern = re.compile(r'\*\*(.*?)\*\*')
#         # Removes the double asterisk pattern and surrounding whitespace
#         self.inline_removal_pattern = re.compile(r'\s*\*\*[^*]*\*\*\s*')

#         # --- Determine Base Save Directory ---
#         self.script_dir = os.path.dirname(self.script_filepath)
#         self.base_save_directory = self.output_dir if self.output_dir else self.script_dir
#         self.script_basename = os.path.splitext(os.path.basename(self.script_filepath))[0]


#     def _log_error(self, message, exception=None):
#         """Helper to log errors and store them."""
#         full_message = f"{message}{f': {exception}' if exception else ''}"
#         logger.error(full_message, exc_info=exception is not None)
#         self.errors.append(full_message)

#     def _initialize_tts(self):
#         """Initializes TTS instances for all required voices."""
#         logger.info("Initializing TTS instances...")
#         required_voices = {details['voice_name'] for details in self.character_voices.values()}

#         for voice_name in required_voices:
#             if voice_name not in self.tts_instances:
#                 try:
#                     # Assumes OpenAITTS class is globally available
#                     self.tts_instances[voice_name] = OpenAITTS(voice=voice_name, model=self.tts_model)
#                     logger.info(f"Initialized TTS for voice '{voice_name}': {self.tts_instances[voice_name]}")
#                 except NameError:
#                     self._log_error("OpenAITTS class definition not found. Cannot initialize TTS.")
#                     return False
#                 except Exception as e:
#                     self._log_error(f"Error initializing TTS instance for '{voice_name}'", e)
#                     return False
#         return True

#     def parse_script(self):
#         """Reads the script file and parses dialogue segments, handling **inline instructions**."""
#         logger.info(f"Parsing script file: {self.script_filepath}")
#         if not os.path.exists(self.script_filepath):
#             self._log_error(f"Script file not found: {self.script_filepath}")
#             return False

#         try:
#             with open(self.script_filepath, 'r', encoding='utf-8') as f:
#                 self.script_content_lines = f.readlines()
#         except Exception as e:
#             self._log_error(f"Error reading script file {self.script_filepath}", e)
#             return False

#         self.dialogue_segments = [] # Reset if called multiple times
#         segment_index = 0
#         for line_num, line in enumerate(self.script_content_lines, 1):
#             match = self.dialogue_pattern.match(line)
#             if match:
#                 character_name = match.group(1).strip()
#                 dialogue_text = match.group(2).strip()

#                 if not dialogue_text: continue # Skip empty

#                 # Find voice config
#                 char_config = self.character_voices.get(character_name, self.character_voices["DEFAULT"])
#                 if character_name not in self.character_voices:
#                      logger.warning(f"Line {line_num}: Character '{character_name}' not explicitly defined. Using default voice.")

#                 base_instructions = char_config["instructions"]
#                 tts_voice_name = char_config["voice_name"]
#                 final_instructions = base_instructions
#                 text_for_tts = dialogue_text
#                 inline_instruction = None

#                 # Check for **inline instructions**
#                 # Use search to find the first occurrence
#                 inline_match = self.inline_instruction_pattern.search(dialogue_text)
#                 if inline_match:
#                     # Extract the adjective/phrase inside **...**
#                     inline_instruction = inline_match.group(1).strip()

#                     # --- Combine instructions (Updated Format) ---
#                     # Prepend the specific instruction for clarity to the TTS
#                     final_instructions = f"For this line, the delivery should be '{inline_instruction}'. General character guidance: {base_instructions}"

#                     # Remove the instruction pattern from the text to be spoken
#                     text_for_tts = self.inline_removal_pattern.sub(' ', dialogue_text).strip()

#                     logger.info(f"Line {line_num}: Found inline instruction '**{inline_instruction}**' for {character_name}. Combined instructions. Cleaned text: '{text_for_tts[:50]}...'")
#                 # else: No inline instruction found, use base_instructions and original dialogue_text

#                 if not text_for_tts:
#                     logger.warning(f"Line {line_num}: Dialogue text became empty after removing instruction for {character_name}. Skipping segment.")
#                     continue

#                 self.dialogue_segments.append(
#                     (segment_index, character_name, text_for_tts, tts_voice_name, final_instructions)
#                 )
#                 segment_index += 1

#         if not self.dialogue_segments:
#             self._log_error("No valid dialogue segments found or parsed in the script.")
#             return False

#         logger.info(f"Parsed {len(self.dialogue_segments)} dialogue segments.")
#         return True

#     # --- Methods _generate_speech_segment_task, generate_segment_audio, ---
#     # --- save_debug_segments, merge_audio, save_final_audio,          ---
#     # --- _display_audio_if_possible, process remain UNCHANGED from     ---
#     # --- the previous class version.                                   ---

#     def _generate_speech_segment_task(self, index, char_name, text, tts_instance_key, instructions):
#         """Task function for generating single audio segment (used by thread pool)."""
#         try:
#             tts_instance = self.tts_instances.get(tts_instance_key)
#             if not tts_instance:
#                  raise ValueError(f"TTS instance for key '{tts_instance_key}' not found.")

#             logger.info(f"Requesting TTS for segment {index} ({char_name}): '{text[:50]}...' using {tts_instance}")

#             # Pass instructions based on instance type and model
#             audio_bytes = None
#             # Check specifically for OpenAITTS and a model known to support instructions well
#             if isinstance(tts_instance, OpenAITTS) and ('gpt-4o' in tts_instance.model or 'tts-1' in tts_instance.model): # Check specific models if needed
#                 # logger.debug(f"Segment {index} instructions: {instructions}")
#                 audio_bytes = tts_instance.tts(text, instructions=instructions)
#                 logger.info(f"Received TTS for segment {index} ({char_name}) with instructions.")
#             else:
#                 if instructions and isinstance(tts_instance, OpenAITTS):
#                     logger.warning(f"Segment {index} ({char_name}): Instructions provided but model {tts_instance.model} might not support them. Calling TTS without.")
#                 elif instructions and not isinstance(tts_instance, OpenAITTS):
#                     logger.warning(f"Segment {index} ({char_name}): TTS instance is not OpenAITTS, instructions ignored.")
#                 # Call the tts method without the 'instructions' keyword argument if not supported
#                 audio_bytes = tts_instance.tts(text)
#                 logger.info(f"Received TTS for segment {index} ({char_name}) without instructions (or model doesn't support them).")


#             # Validation
#             if not audio_bytes or len(audio_bytes) < 100: # Check for empty or too small
#                 logger.warning(f"TTS returned invalid/empty audio for segment {index} ({char_name}). Size={len(audio_bytes) if audio_bytes else 0}. Skipping.")
#                 return index, None
#             return index, audio_bytes

#         except Exception as e:
#             self._log_error(f"Error in TTS task for segment {index} ({char_name})", e)
#             return index, None # Return index and None on error

#     def generate_segment_audio(self):
#         """Generates audio for all parsed dialogue segments concurrently."""
#         if not self.dialogue_segments:
#             self._log_error("Cannot generate audio, no dialogue segments parsed.")
#             return False
#         if not self.tts_instances:
#              self._log_error("Cannot generate audio, TTS instances not initialized.")
#              return False

#         logger.info(f"Generating audio for {len(self.dialogue_segments)} segments using up to {self.max_workers} workers...")
#         self.raw_audio_results = {} # Reset results
#         futures_to_index = {}

#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             for index, char_name, text_to_speak, tts_instance_key, combined_instructions in self.dialogue_segments:
#                 future = executor.submit(
#                     self._generate_speech_segment_task,
#                     index, char_name, text_to_speak, tts_instance_key, combined_instructions
#                 )
#                 futures_to_index[future] = index

#             for future in concurrent.futures.as_completed(futures_to_index):
#                 original_index = futures_to_index[future]
#                 try:
#                     idx, audio_data = future.result()
#                     if audio_data:
#                         self.raw_audio_results[idx] = audio_data
#                     # else: Error/Warning already logged in _generate_speech_segment_task
#                 except Exception as e:
#                      # This catches errors during future.result() itself, unlikely if task handles errors
#                     self._log_error(f"Error retrieving result for segment {original_index}", e)

#         if not self.raw_audio_results:
#             logger.warning("No audio segments were successfully generated.")
#             return len(self.raw_audio_results) > 0 # Return False if zero segments succeeded

#         logger.info(f"Successfully generated audio for {len(self.raw_audio_results)} out of {len(self.dialogue_segments)} segments.")
#         return True

#     def save_debug_segments(self):
#         """Saves successfully generated individual audio segments to disk."""
#         if not self.raw_audio_results:
#             logger.info("No raw audio results to save as debug segments.")
#             return False

#         self.debug_segments_path = os.path.join(self.base_save_directory, "_debug_audio_segments")
#         logger.info(f"Saving {len(self.raw_audio_results)} individual audio segments for debugging to: {self.debug_segments_path}")

#         try:
#             os.makedirs(self.debug_segments_path, exist_ok=True)

#             for index, audio_bytes in self.raw_audio_results.items():
#                 if not audio_bytes: continue

#                 try:
#                     character_name = self.dialogue_segments[index][1]
#                     safe_char_name = character_name.replace(' ', '_').strip()
#                 except (IndexError, TypeError) as e:
#                     self._log_error(f"Error getting character name for debug segment index {index}", e)
#                     safe_char_name = "UnknownChar"

#                 segment_filename = f"segment_{index:03d}_{safe_char_name}.mp3"
#                 segment_filepath = os.path.join(self.debug_segments_path, segment_filename)

#                 try:
#                     with open(segment_filepath, 'wb') as seg_file:
#                         seg_file.write(audio_bytes)
#                 except IOError as e:
#                     self._log_error(f"Failed to save debug segment {segment_filepath}", e)

#             return True

#         except Exception as e:
#             self._log_error(f"Could not create or write to debug directory {self.debug_segments_path}", e)
#             return False


#     def merge_audio(self, use_ffmpeg=True):
#         """Merges the generated audio segments in order."""
#         if not self.raw_audio_results:
#             self._log_error("No audio segments available to merge.")
#             return False

#         logger.info("Preparing to merge audio segments...")
#         self.ordered_audio_bytes = []
#         missing_count = 0
#         for i in range(len(self.dialogue_segments)):
#             audio_segment = self.raw_audio_results.get(i)
#             if audio_segment:
#                 self.ordered_audio_bytes.append(audio_segment)
#             else:
#                 missing_count += 1

#         if missing_count > 0:
#              logger.warning(f"{missing_count} audio segments were missing or invalid and will be skipped in the final merge.")

#         if not self.ordered_audio_bytes:
#             self._log_error("No valid audio segments remained after ordering. Cannot merge.")
#             return False

#         logger.info(f"Merging {len(self.ordered_audio_bytes)} audio segments in order...")
#         try:
#             # Assumes merge_mp3s function is globally available
#             self.merged_audio = merge_mp3s(self.ordered_audio_bytes, use_ffmpeg=use_ffmpeg)
#             if self.merged_audio:
#                  logger.info("Audio segments merged successfully.")
#                  return True
#             else:
#                  self._log_error("Merging process returned no data.")
#                  return False
#         except NameError:
#             self._log_error("merge_mp3s function definition not found. Cannot merge audio.")
#             return False
#         except Exception as e:
#             self._log_error("Error during merging process", e)
#             return False

#     def save_final_audio(self):
#         """Saves the merged audio to the final output file."""
#         if not self.merged_audio:
#             self._log_error("No merged audio data available to save.")
#             return None

#         output_filename = f"{self.script_basename}_audio.mp3"
#         self.final_output_path = os.path.join(self.base_save_directory, output_filename)

#         logger.info(f"Saving final merged audio to: {self.final_output_path}")
#         try:
#             os.makedirs(self.base_save_directory, exist_ok=True)
#             with open(self.final_output_path, 'wb') as f_out:
#                 f_out.write(self.merged_audio)
#             logger.info("Final audio saved successfully.")
#             self._display_audio_if_possible(self.final_output_path)
#             return self.final_output_path
#         except Exception as e:
#             self._log_error(f"Error saving merged audio file to {self.final_output_path}", e)
#             self.final_output_path = None
#             return None

#     def _display_audio_if_possible(self, audio_path):
#         """Try displaying audio in IPython if available."""
#         try:
#             # Check if running in an IPython environment
#             shell = get_ipython().__class__.__name__
#             if shell == 'ZMQInteractiveShell': # Jupyter notebook or qtconsole
#                  from IPython.display import display, Audio
#                  display(Audio(audio_path))
#             # Add other shell types if needed, e.g., 'TerminalInteractiveShell' for ipython terminal
#         except NameError:
#              pass # Not in IPython
#         except Exception as e:
#              logger.debug(f"Could not display audio in IPython: {e}")


#     def process(self, save_debug=True, merge_with_ffmpeg=True):
#         """Runs the full audio generation pipeline."""
#         self.errors = []

#         if not self._initialize_tts():
#             return None

#         if not self.parse_script():
#             return None

#         # Check if generation failed completely
#         if not self.generate_segment_audio() and not self.raw_audio_results:
#             logger.error("Audio generation failed completely.")
#             return None

#         if save_debug:
#             self.save_debug_segments()

#         if not self.merge_audio(use_ffmpeg=merge_with_ffmpeg):
#             logger.error("Audio merging failed. Check errors and debug segments if saved.")
#             return None

#         final_path = self.save_final_audio()

#         if not final_path and self.errors:
#              logger.error("Processing finished with errors. Final audio not saved.")
#         elif not final_path:
#              logger.error("Processing finished, but final audio could not be saved.")
#         else:
#              logger.info("Processing finished successfully.")

#         return final_path


# %%
# script_file = '/Users/jong/Documents/PodcastGPT/tv_show_workspaces/20250330_191704/Seinfeld_hilarious_Seinfeld_episode_abo/final_script_3acts.txt'

# # Instantiate the generator
# generator = SeinfeldAudioGenerator(script_file) # , output_dir='/custom/output/path' if needed)

# # Run the process
# final_audio_path = generator.process(save_debug=True, merge_with_ffmpeg=True)

# if final_audio_path:
#     print(f"Success! Final audio saved to: {final_audio_path}")
# else:
#     print("Processing failed or did not complete.")
#     print("Check generator state for details:")
#     print(f"  - Parsed Segments: {len(generator.dialogue_segments)}")
#     print(f"  - Generated Audio Segments: {len(generator.raw_audio_results)}")
#     if generator.debug_segments_path:
#         print(f"  - Debug Segments saved to: {generator.debug_segments_path}")
#     if generator.ordered_audio_bytes and not generator.merged_audio:
#         print(f"  - Merging failed, but {len(generator.ordered_audio_bytes)} ordered segments exist.")
#     print(f"  - Errors encountered: {len(generator.errors)}")
#     for i, err in enumerate(generator.errors):
#         print(f"    Error {i+1}: {err}")


# %%
