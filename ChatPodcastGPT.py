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
# import vertexai
# import vertexai.preview.generative_models
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
logger = jonlog.getLogger()
openai.api_key = os.environ.get("OPENAI_KEY", None) or open('/Users/jong/.openai_key').read().strip()


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
    def tts(self, text):
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=self._voice_id,
            Engine="neural",
        )
        # The audio stream containing the synthesized speech
        audio_stream = response.get('AudioStream')
        return audio_stream.read()


# %%
class AWSChat:
    MODELS = {
        "claude-instant": "anthropic.claude-instant-v1",
        "claude-best": "anthropic.claude-v2:1",
    }

    @classmethod
    def msg(cls, messages=None, model="anthropic.claude-instant-v1", **kwargs):
        client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        try:
            # The different model providers have individual request and response formats.
            # For the format, ranges, and default values for Anthropic Claude, refer to:
            # https://docs.anthropic.com/claude/reference/complete_post

            # Claude requires you to enclose the prompt as follows:
            # enclosed_prompt = "Human: " + prompt + "\n\nAssistant:"
            prompt = "\n\n".join(
                [f'{"" if (msg["role"] == "system" and model == cls.MODELS["claude-best"]) else ("Human" if msg["role"] != "assistant" else "Assistant")}: {msg["content"]}' for msg in messages] +
                ["Assistant:"]
            )

            if 'temperature' not in kwargs:
                kwargs['temperature'] = 1
            body = {
                "prompt": prompt,
                "max_tokens_to_sample": 2048,
                **kwargs
            }
            response = client.invoke_model(
                modelId=model, body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            completion = response_body["completion"]
            return completion
        except ClientError as e:
            logger.exception(f"Couldn't invoke {model}", e)
            raise


# %%
# class GoogleChat:
#     MODELS = {
#         "gemini-pro": "gemini-pro",
#     }

#     @classmethod
#     def consolidate_messages(cls, message_list):
#         if not message_list:
#             return []
    
#         consolidated = []
#         current_role = None
#         current_content = ""
    
#         for message in message_list:
#             role = message.get("role")
#             content = message.get("content", "")
    
#             if role == "system":
#                 role = "user"
#             if role == current_role:
#                 current_content += "\n" + content
#             else:
#                 if current_role is not None:
#                     consolidated.append({"role": current_role, "content": current_content})
#                 current_content = content
#                 current_role = role
    
#         if current_role is not None:
#             consolidated.append({"role": current_role, "content": current_content})
    
#         return consolidated

#     @classmethod
#     def msg(cls, messages=None, model="gemini-pro", **kwargs):
#         vertexai.init(project='summer2023-392312', location='us-central1')
#         model = vertexai.preview.generative_models.GenerativeModel(model)
#         contents = [vertexai.generative_models._generative_models.Content(
#             role="user" if msg["role"] != "assistant" else "model",
#             parts=[vertexai.generative_models._generative_models.Part.from_text(msg["content"])]
#         ) for msg in cls.consolidate_messages(messages)]
#         response = model.generate_content(contents=contents)
#         return response.text

# %%
class MistralChat:
    MODELS = {
        "mistral-medium": "mistral-medium",
        "mistral-small": "mistral-small",
    }
    api_key = os.environ.get("MISTRAL_API_KEY") or open('/Users/jong/.mistral_apikey').read().strip()

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
        client = MistralClient(api_key=cls.api_key)
        if not any(msg['role'] == 'user' for msg in messages):
            messages[-1]['role'] = 'user'
        chat_response = client.chat(
            model=model,
            messages=[MistralChatMessage(**msg) for msg in cls.consolidate_messages(messages)],
        )
        return chat_response.choices[0].message.content


# %%
DEFAULT_MODEL = 'gpt-4-1106-preview'
DEFAULT_LENGTH  = 80_000

class Chat:
    class Model(enum.Enum):
        GPT3_5 = "gpt-3.5-turbo"
        GPT_4  = "gpt-4-1106-preview"

    def __init__(self, system, max_length=DEFAULT_LENGTH):
        self._system = system
        self._max_length = max_length
        self._history = [
            {"role": "system", "content": self._system},
        ]

    @classmethod
    def num_tokens_from_text(cls, text, model=DEFAULT_MODEL):
        """Returns the number of tokens used by some text."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @classmethod
    def num_tokens_from_messages(cls, messages, model=DEFAULT_MODEL):
        """Returns the number of tokens used by a list of messages."""
        encoding = tiktoken.encoding_for_model(model)
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
    def _msg(self, *args, model=DEFAULT_MODEL, **kwargs):
        logger.info(f'requesting chatcompletion {model=}...')
        if model.startswith("AWS/"):
            model = model[4:]
            resp = AWSChat.msg(
                messages=self._history,
                **kwargs
            )
        # elif model.startswith("GOOGLE/"):
        #     model = model[7:]
        #     resp = GoogleChat.msg(messages=self._history, model=model, **kwargs)
        elif model.startswith("MISTRAL/"):
            model = model[8:]
            resp = MistralChat.msg(messages=self._history, model=model, **kwargs)
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
    def __init__(self, topic, podcast="award winning", max_length=DEFAULT_LENGTH, hosts=['Tom', 'Jen'], host_voices=[AWSPollyTTS(AWSPollyTTS.MAN), AWSPollyTTS(AWSPollyTTS.WOMAN)], extra_system=None):
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
            
    def step(self, msg=None, skip_aud=False, ret_aud=True, **kwargs):
        msg = self.message(msg, **kwargs)
        if skip_aud: return msg
        aud = self.text2speech(msg)
        if ret_aud: return msg, aud
        return msg


# %%
class PodcastRSSFeed:
    """Class to handle rss feed operations using github pages."""

    def __init__(self, org, repo, xml_path):
        self.org = org
        self.repo = repo
        self.xml_path = xml_path
        self.local_xml_path = self.download_podcast_xml()

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
# # %%time
# ep = Episode(
#     episode_type='narration',
#     topic="Hidden History: Unraveling 3 of History's Funniest Mysteries",
#     max_length=10_000,
#     # text_model='gpt-4-1106-preview',
#     text_model='MISTRAL/mistral-medium',
# )
# outline, txt = ep.step(nparts='3')

# %%
# print(txt)

# %%
# print(txt)
