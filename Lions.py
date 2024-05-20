# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import imaplib
import smtplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime, formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import bs4
import re
import functools
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import datetime
from lxml import etree as ET
import pytz
import html
import random
import os
import google.generativeai as genai
import tempfile

import jonlog

logger = jonlog.getLogger()
logger.info("Reloaded :)")


# -

class ZohoMail:
    def __init__(self, username, password):
        self._mail = imaplib.IMAP4_SSL('imap.zoho.com')
        self._smtp = smtplib.SMTP_SSL('smtp.zoho.com', 465)
        self._username = username
        self._password = password
        self.login()

    def login(self):
        self._smtp.login(self._username, self._password)
        self._mail.login(self._username, self._password)
        self._mail.select('INBOX')

    def search_emails(self):
        # 700lions@gmail.com
        resp, data = self._mail.search(None, 'unseen FROM "700lions@gmail.com"')
        return data[0].split()

    @functools.cache
    def get_msg(self, msg_id):
        _status, msg_data = self._mail.uid('fetch', msg_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        return msg

    def get_email_parts(self, msg):
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            yield part

    def get_email_data(self, msg_id):
        msg = self.get_msg(msg_id)
        sender = decode_header(msg.get("From"))[0][0]
        if isinstance(sender, bytes):
            sender = sender.decode()
        subject = decode_header(msg.get("Subject"))[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        date = parsedate_to_datetime(msg.get("Date"))

        text = ''
        attachments = []
        for part in self.get_email_parts(msg):
            if part.get_content_type().startswith('text'):
                if part.get_content_type().endswith('/html'):
                    soup = bs4.BeautifulSoup(part.get_payload(decode=True).decode(), 'html.parser')
                    text = soup.get_text()
                else:
                    text = part.get_payload(decode=True).decode()
            else:
                attachments.append({"data": part.get_payload(decode=True), "type": part.get_content_type()})

        return {'id': msg_id, 'sender': sender, 'subject': subject, 'date': date, 'text': text, 'attachments': attachments}

    def get(self, limit=10):
        msg_ids = self.search_emails()[-limit:]
        emails = [self.get_email_data(msg_id) for msg_id in msg_ids]
        return emails

    def mark_read(self, msg_id):
        self._mail.uid('store', msg_id, '+FLAGS', '(\Seen)')

    def forward_email(self, msg_id, forward_text):
        original_msg = self.get_msg(msg_id)
        tag = forward_text.split('] ')[0] + ']'
        
        # Create a new MIMEMultipart message for forwarding
        msg = MIMEMultipart()
        msg['From'] = '700lions@zohomail.com'
        msg['Subject'] = f"[Gemini] {tag} Fwd: {original_msg['Subject']}"
        recipients = [original_msg['From']] + original_msg.get_all('Cc', [])
        if 'worker' not in tag.lower():
            recipients += ['jonathanallengrant@gmail.com']
    
        # Attach the forward text
        msg.attach(MIMEText(forward_text, 'plain'))
        
        # Attach any attachments from the original message
        for part in original_msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            attachment = MIMEBase(part.get_content_maintype(), part.get_content_subtype())
            attachment.set_payload(part.get_payload(decode=True))
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', part.get('Content-Disposition'))
            msg.attach(attachment)
    
        # Send the forwarded email
        self._smtp.sendmail(msg['From'], recipients, msg.as_string())


"""
Gemini API
"""
class GoogleAI:
    @classmethod
    def get_api_key(cls):
        return os.environ.get("GEMINI_API_KEY") or open('/Users/jong/.gemini_apikey').read().strip()

    def __init__(self, model='gemini-1.5-flash-latest', tags=['Boring', 'Animal', 'Worker', 'Non-Worker Person', 'Unknown']):
        self.model = model
        self.tags = tags
        genai.configure(api_key=self.get_api_key())

    def upload_to_gemini(self, path, mime_type=None):
      """Uploads the given file to Gemini.
    
      See https://ai.google.dev/gemini-api/docs/prompting_with_media
      """
      file = genai.upload_file(path, mime_type=mime_type)
      print(f"Uploaded file '{file.display_name}' as: {file.uri}")
      return file

    def message(self, photos):
        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        adjective = random.choice(["sarcastic", "exhilarated", "infuriated", "melancholic", "bewildered", "ecstatic", "aghast", "enthralled", "despondent", "euphoric", "apoplectic", "wistful", "overjoyed", "crestfallen", "flabbergasted", "elated", "exasperated", "jubilant", "livid", "pensive", "thrilled", "evil", "depressed"])
        person = random.choice(["Donald Trump campaign rant", "Yoda", "Darth Vader", "Einstein", "Elon Musk Tweet", "Shakespearean monologue", "Martin Luther King Jr. speech", "Winston Churchill wartime address", "Gordon Ramsay critique", "Neil deGrasse Tyson lecture", "Steve Jobs keynote", "Joe Rogan podcast", "Larry David rant", "Jerry Seinfeld standup"])
        system = f'At the start of your message, tag the photo with just one of these tags, surrounded by square brackets. [{", ".join(self.tags)}]. Always respond truthfully. However, respond in the style of {adjective} {person}'
        logger.info(system)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        # TODO Make these files available on the local file system
        # You may need to update the file paths
        #TODO: Save photo to disk tmp
        image_drives = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, photo in enumerate(photos):
                photo_suffix = photo['type'].split('/')[-1]
                photo_path = f"{tmpdir}/{i}.{photo_suffix}"
                with open(photo_path, 'wb') as f:
                    f.write(photo['data'])
                image_drives.append(self.upload_to_gemini(photo_path, mime_type=photo['type']))

        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": image_drives}]
        )
        response = chat_session.send_message(f"Here are photo(s) from my security camera. What do you see? At the start of your message, tag the photo with just one of these tags, surrounded by square brackets. {self.tags}")
        return f"{adjective} {person}", response.text.strip()


mail = ZohoMail('700lions@zoho.com', 'tnb2xeh!PXV4vrc.xza')


def do_email(mail, email_data):
    if len(m.get('attachments', [])) == 0:
        mail.mark_read(email_data['id'])
        return
    chat = GoogleAI()
    persona, resp = chat.message(email_data['attachments'])
    mail.forward_email(email_data['id'], f"{resp}\n\nFrom: {persona} (gemini-1.5-flash)")
    mail.mark_read(email_data['id'])


for m in mail.get(limit=10):
    do_email(mail, m)






