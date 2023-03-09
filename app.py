from flask import Flask, request, abort

import os
import json
import openai
import tiktoken
import structlog
import cachetools
logger = structlog.getLogger()
openai.api_key = os.environ["OPENAI_KEY"]


class Chat:
    def __init__(self, topic, podcast=None, max_length=4096//4, hosts=['Tom', 'Jen']):
        if podcast is None:
            podcast = "award winning NPR"
        system = f"You are an {podcast} podcast with hosts {hosts[0]} and {hosts[1]}."
        self._system = system
        self._topic = topic
        self._max_length = max_length
        self._hosts = hosts
        self._history = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": f"Generate a podcast episode about {topic}, including history and other fun facts. Reference published scientific journals."},
        ]

    @classmethod
    def num_tokens_from_messages(cls, messages, model="gpt-3.5-turbo"):
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
        
    def message(self, next_msg=None):
        # TODO: Optimize this if slow through easy caching
        while len(self._history) > 1 and self.num_tokens_from_messages(self._history) > self._max_length:
            logger.info(f'Popping message: {self._history.pop(1)}')
        if next_msg is not None:
            self._history.append({"role": "user", "content": next_msg})
        logger.info('requesting openai...')
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self._history,
        )
        logger.info('received openai...')
        text = resp.choices[0].message.content
        self._history.append({"role": "assistant", "content": text})
        return text

app = Flask(__name__)
from flask_sslify import SSLify
if 'DYNO' in os.environ: # only trigger SSLify if the app is running on Heroku
    sslify = SSLify(app)

LRU_MAX = 420_69
chat_cache = cachetools.LRUCache(LRU_MAX)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/chat/<chat_id>", methods=['POST'])
def chat(chat_id):
    data = request.form
    if chat_id not in chat_cache:
        chat_cache[chat_id] = Chat(data['topic'], podcast=data.get('podcast'))
    msg = data.get('msg')
    text = chat_cache[chat_id].message(next_msg=msg)
    return text

@app.route("/history/<chat_id>", methods=['GET'])
def history(chat_id):
    if chat_id not in chat_cache:
        abort(404)
    return json.dumps(chat_cache[chat_id]._history)
