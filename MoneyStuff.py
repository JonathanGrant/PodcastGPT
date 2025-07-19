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
import requests
import datetime
import re
import random
import os
import tempfile
from urllib.parse import urljoin
from bs4 import BeautifulSoup as BS
import jonlog
from ChatPodcastGPT import *

logger = jonlog.getLogger()

# +
import os
import re
import mimetypes
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4

BASE_URL = "https://overcast.fm"
LOGIN_PATH = "/login"
UPLOADS_PATH = "/uploads"
UPLOAD_SUCCEEDED_PATH = "/podcasts/upload_succeeded"
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)


class OvercastError(Exception):
    pass


class OvercastLoginError(OvercastError):
    pass


class OvercastUploadError(OvercastError):
    pass


class OvercastUploader:
    """
    Single-purpose Overcast uploader.

    Usage:
        uploader = OvercastUploader(email, password)
        result = uploader.upload("/path/to/audio.wav")

    result =>
        {
          'key': '3895085/inbox/My Episode.wav',
          'etag': '83c8942a1775ca1405d39b8772fe7f42',
          'location': 'https://uploads-overcast.s3.amazonaws.com/...encoded...',
          'size_bytes': 12345678
        }
    """

    def __init__(
        self,
        email: str,
        password: str,
        *,
        timeout: float = 30.0,
        verify: bool = True,
        user_agent: str = DEFAULT_UA,
    ):
        self.email = email
        self.password = password
        self.timeout = timeout
        self.verify = verify
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": BASE_URL,
            "Referer": BASE_URL + LOGIN_PATH,
        })
        self._login()

    # --------------------------- INTERNAL HELPERS --------------------------- #

    def _login(self):
        """Perform login with optional CSRF token extraction."""
        # GET login page to grab CSRF/authenticity token (if any)
        r = self.session.get(BASE_URL + LOGIN_PATH,
                             timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        token = self._extract_csrf_token(r.text)

        form = {
            "email": self.email,
            "password": self.password,
        }
        if token:
            # Adjust name if Overcast uses a specific one; keep both tries.
            form["authenticity_token"] = token

        pr = self.session.post(
            BASE_URL + LOGIN_PATH,
            data=form,
            timeout=self.timeout,
            verify=self.verify,
            allow_redirects=False,
        )

        if pr.status_code in (301, 302, 303, 307, 308):
            # success redirect
            return

        if pr.status_code != 200:
            raise OvercastLoginError(f"Login failed (status={pr.status_code})")

        if re.search(r"(invalid|incorrect)", pr.text, re.I):
            raise OvercastLoginError("Login failed: invalid credentials.")

    @staticmethod
    def _extract_csrf_token(html: str) -> Optional[str]:
        # Look for common token field names
        m = re.search(r'name="(?:authenticity_token|csrf_token)" value="([^"]+)"', html)
        return m.group(1) if m else None

    def _fetch_upload_form(self) -> Dict[str, str]:
        """
        GET /uploads and parse the AWS S3 form fields and action URL.

        Returns a dict:
            {
              'action': 'https://uploads-overcast.s3.amazonaws.com/',
              'fields': { <form field name>: <value>, ... }
            }
        """
        r = self.session.get(BASE_URL + UPLOADS_PATH,
                             timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Find the form posting to S3 (action contains 's3')
        form = None
        for f in soup.find_all("form"):
            action = (f.get("action") or "").lower()
            if "s3" in action and "amazonaws.com" in action:
                form = f
                break
        if not form:
            raise OvercastUploadError("Could not locate S3 upload form on /uploads page.")

        action_url = form.get("action")
        fields = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            # For file input we skip; `requests` will supply file part.
            if inp.get("type") == "file":
                continue
            value = inp.get("value", "")
            fields[name] = value

        return {"action": action_url, "fields": fields}

    def _notify_success(self, key: str):
        """POST key to upload_succeeded endpoint."""
        r = self.session.post(
            BASE_URL + UPLOAD_SUCCEEDED_PATH,
            data={"key": key},
            timeout=self.timeout,
            verify=self.verify,
        )
        if r.status_code not in (200, 204, 302):
            raise OvercastUploadError(
                f"Upload success notification failed (status={r.status_code})"
            )

    # --------------------------- PUBLIC API --------------------------------- #

    def upload(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, str]:
        """
        Perform the full Overcast upload.

        Steps:
            1. Fetch /uploads & parse S3 form (policy, signature, key template, etc.).
            2. Replace the 'key' field's filename segment with display_name or file basename.
            3. Multipart POST to S3.
            4. Notify Overcast.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        form_info = self._fetch_upload_form()
        action_url = form_info["action"]
        fields = form_info["fields"]

        if "key" not in fields:
            raise OvercastUploadError("'key' field missing in parsed upload form.")

        basename = display_name or os.path.basename(file_path)
        # Overcast typical key format is like: 3895085/inbox/${filename}
        # Replace only the tail after last '/' with our basename.
        original_key = fields["key"]
        key_parts = original_key.split("/")
        key_parts[-1] = basename
        final_key = "/".join(key_parts)
        fields["key"] = final_key

        # If content-type not enforced by policy, it's okay; else include.
        mime = mimetypes.guess_type(basename)[0] or "audio/wav"

        files = {
            # Field name usually "file" for S3 forms
            "file": (basename, open(file_path, "rb"), mime)
        }

        # Perform S3 upload
        resp = self.session.post(
            action_url,
            data=fields,
            files=files,
            timeout=self.timeout,
            verify=self.verify,
        )
        # Close file handle (requests keeps a reference)
        files["file"][1].close()

        if resp.status_code not in (200, 201, 204):
            snippet = resp.text[:400]
            raise OvercastUploadError(
                f"S3 upload failed status={resp.status_code} body_snippet={snippet}"
            )

        etag = resp.headers.get("ETag", "").strip('"')
        location = resp.headers.get("Location") or f"{action_url.rstrip('/')}/{final_key}"

        # Notify Overcast UI
        self._notify_success(final_key)

        return {
            "key": final_key,
            "etag": etag,
            "location": location,
            "size_bytes": os.path.getsize(file_path),
        }



# -

def clean_text(txt):
    txt = txt.replace('\r', '\n').replace('\xa0', ' ')
    txt = re.sub('\[\d\]', '', txt)
    return txt


# +
import requests, re, html, time
from bs4 import BeautifulSoup as BS
from urllib.parse import urljoin
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable

# ------------------ Utility ------------------ #

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

def clean_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s.replace('\xa0', ' ').replace('\r', ' ')).strip()

def is_recent_relative(text: str, max_days: int = 5) -> bool:
    """
    Accept strings like '2 days ago', '1 day ago', '5 days ago', '3 hours ago'.
    """
    m = re.match(r'(\d+)\s+(day|days|hour|hours)\s+ago', text.lower())
    if not m:
        return False
    num, unit = int(m[1]), m[2]
    if 'hour' in unit:
        return num <= 24 and max_days >= 1
    if 'day' in unit:
        return num <= max_days
    return False

# ------------------ Data Structures ------------------ #

@dataclass
class ArticleMeta:
    title: str
    link: str
    date_text: Optional[str] = None

@dataclass
class Section:
    heading: Optional[str]
    paragraphs: List[str] = field(default_factory=list)
    list_items: List[str] = field(default_factory=list)
    blockquotes: List[str] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        out = []
        if self.heading:
            out.append(f"## {self.heading}")
        out.extend(self.paragraphs)
        if self.list_items:
            out += [f"- {li}" for li in self.list_items]
        for bq in self.blockquotes:
            out.append("> " + bq.replace('\n', '\n> '))
        if self.footnotes:
            out.append("\n".join(f"[{i+1}] {f}" for i, f in enumerate(self.footnotes)))
        return "\n\n".join(out)

@dataclass
class ArticleContent:
    url: str
    title: str
    raw_lines: List[str]
    sections: List[Section]
    markdown: str

# ------------------ Core Class ------------------ #

class MoneyStuff:
    BASE_URL = 'https://newsletterhunt.com/'
    LIST_URL = urljoin(BASE_URL, 'newsletters/money-stuff-by-matt-levine')

    def __init__(self, session: Optional[requests.Session] = None, parser_priority: Iterable[str] = ('lxml', 'html5lib', 'html.parser')):
        self.session = session or requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.parser_priority = parser_priority
        self._cache: Dict[str, ArticleContent] = {}

    # ---------- HTTP helpers ---------- #
    def _get(self, url: str, retries: int = 3, backoff: float = 0.75) -> requests.Response:
        last_exc = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, timeout=20)
                r.raise_for_status()
                return r
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff * (2 ** attempt))
        raise RuntimeError(f"Failed to GET {url}: {last_exc}")

    def list(self, max_days: int = 5, limit: Optional[int] = None) -> List[ArticleMeta]:
        r = self._get(self.LIST_URL)
        soup = BS(r.text, 'html.parser')  # list page is simple
        articles = []
        for art in soup.find_all('article'):
            h2 = art.find('h2')
            t = art.find('time')
            a = h2.find_parent('a') if h2 else None
            if not (h2 and t and a and a.get('href')):
                continue
            date_text = t.get_text(strip=True)
            if is_recent_relative(date_text, max_days=max_days):
                articles.append(ArticleMeta(
                    title=clean_ws(h2.get_text()),
                    link=a['href'],
                    date_text=date_text
                ))
            if limit and len(articles) >= limit:
                break
        return articles

    # ---------- Parsing pipeline ---------- #

    def get(self, article: ArticleMeta | Dict) -> ArticleContent:
        # Accept dict or ArticleMeta
        if isinstance(article, dict):
            article = ArticleMeta(**article)

        full_url = urljoin(self.BASE_URL, article.link)
        if full_url in self._cache:
            return self._cache[full_url]

        page = self._get(full_url).text
        page_soup = BS(page, 'html.parser')  # outer page simple enough
        iframe = page_soup.find('iframe', id='iframeEmail')
        if not iframe:
            raise ValueError(f"iframeEmail not found: {full_url}")

        fragment = self._extract_iframe_fragment(iframe, full_url)
        body_soup = self._parse_fragment(fragment)

        raw_lines = self._extract_lines(body_soup)
        sections = self._structure_sections(body_soup)
        markdown = self._sections_to_markdown(sections, article.title, full_url)

        content = ArticleContent(
            url=full_url,
            title=article.title,
            raw_lines=raw_lines,
            sections=sections,
            markdown=markdown
        )
        self._cache[full_url] = content
        return content

    # ---------- Internal helpers ---------- #

    def _extract_iframe_fragment(self, iframe, base_url: str) -> str:
        # Prefer srcdoc
        srcdoc = iframe.get('srcdoc')
        if srcdoc:
            # Unescape twice if needed
            first = html.unescape(srcdoc)
            second = html.unescape(first)
            fragment = second if second != first else first
            return fragment.strip()
        # Fallback to src URL
        src = iframe.get('src')
        if not src:
            raise ValueError(f"No srcdoc or src on iframe at {base_url}")
        iframe_url = urljoin(base_url, src)
        return self._get(iframe_url).text

    def _parse_fragment(self, fragment: str) -> BS:
        # Try parsers in priority
        for parser in self.parser_priority:
            try:
                soup = BS(fragment, parser)
                break
            except Exception:
                continue
        body = soup.find('body')
        if body is None:
            # Wrap manually
            wrapped = f"<body>{fragment}</body>"
            for parser in self.parser_priority:
                try:
                    soup = BS(wrapped, parser)
                    break
                except Exception:
                    continue
        return soup.body or soup  # fallback to root

    def _extract_lines(self, body: BS) -> List[str]:
        text = body.get_text(separator='\n', strip=True)
        lines = []
        for line in text.split('\n'):
            line = clean_ws(line)
            if not line:
                continue
            if self._is_boilerplate(line):
                continue
            lines.append(line)
        return lines

    def _is_boilerplate(self, line: str) -> bool:
        low = line.lower()
        if low.startswith('view in browser'):
            return True
        if low.startswith('follow us'):
            return True
        if 'you received this message because' in low:
            return True
        return False

    def _structure_sections(self, body: BS) -> List[Section]:
        sections: List[Section] = []
        current = Section(heading=None)

        def flush():
            nonlocal current
            if current.heading or current.paragraphs or current.list_items or current.blockquotes or current.footnotes:
                sections.append(current)
            current = Section(heading=None)

        # Footnotes often in div#footnote-x
        footnote_map: Dict[str, str] = {}
        for fdiv in body.select('div[id^=footnote-]'):
            fid = fdiv.get('id')
            txt = clean_ws(fdiv.get_text(separator=' ', strip=True))
            if txt:
                footnote_map[fid] = re.sub(r'^\[\d+\]\s*', '', txt)

        # Iterate top-level tables / headings / paragraphs
        for el in body.descendants:
            if getattr(el, 'name', None) == 'h2':
                flush()
                current.heading = clean_ws(el.get_text())
            elif getattr(el, 'name', None) == 'p':
                txt = clean_ws(el.get_text())
                if txt and not self._is_boilerplate(txt):
                    current.paragraphs.append(txt)
            elif getattr(el, 'name', None) in ('li',):
                txt = clean_ws(el.get_text())
                if txt and not self._is_boilerplate(txt):
                    current.list_items.append(txt)
            elif getattr(el, 'name', None) == 'blockquote':
                btxt = clean_ws(el.get_text(separator=' '))
                if btxt:
                    current.blockquotes.append(btxt)
        flush()

        # Attach footnotes at end section (or create one)
        if footnote_map:
            # Simple heuristic: put all footnotes into a terminal section
            foot_section = Section(heading="Footnotes",
                                   footnotes=[footnote_map[k] for k in sorted(footnote_map)])
            sections.append(foot_section)
        return sections

    def _sections_to_markdown(self, sections: List[Section], title: str, url: str) -> str:
        out = [f"# {title}", f"*Source: {url}*\n"]
        for s in sections:
            out.append(s.to_markdown())
        return "\n\n".join(filter(None, out)).strip()


# +
import io, os, tempfile, subprocess, shutil, logging
from typing import List
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def merge_wav_pydub(wav_bytes_list: List[bytes]) -> bytes:
    """
    Simple decode + append + re-encode to WAV via pydub/ffmpeg or audioop.
    Loses original exact headers but fine for most uses.
    """
    if not wav_bytes_list:
        return b""
    combined = AudioSegment.from_file(io.BytesIO(wav_bytes_list[0]), format="wav")
    for b in wav_bytes_list[1:]:
        seg = AudioSegment.from_file(io.BytesIO(b), format="wav")
        combined += seg
    out = io.BytesIO()
    combined.export(out, format="wav")
    return out.getvalue()

def merge_wav(wav_bytes_list: List[bytes], use_ffmpeg: bool = True, strict: bool = False) -> bytes:
    """
    Merge multiple WAV byte blobs into one WAV (concatenated sequentially).

    Strategy:
      1. (Optional) ffmpeg concat demuxer using stream copy (-c copy) -> NO re-encode (fast, lossless)
         REQUIREMENT: All input WAVs must share identical format (sample rate, channels, bit depth).
      2. Fallback: decode+append+re-encode with pydub (slower, but robust to differing params).

    Args:
        wav_bytes_list : list of WAV byte strings.
        use_ffmpeg     : attempt ffmpeg first if available.
        strict         : if True, raise instead of falling back when ffmpeg fails or formats differ.

    Returns:
        bytes of merged WAV (empty bytes if input list empty).

    Notes:
        - If you frequently merge many short clips: pre-normalize them to one format to keep fast path.
        - For very large merges, a streaming writer (wave + manual frame concat) would be more memory efficient.
    """
    if not wav_bytes_list:
        logger.warning("Empty WAV list.")
        return b""

    valid = [b for b in wav_bytes_list if isinstance(b, bytes) and len(b) > 100]
    if not valid:
        logger.warning("No sufficiently large WAV segments.")
        return b""

    ffmpeg_path = shutil.which("ffmpeg") if use_ffmpeg else None
    if ffmpeg_path:
        tmp_inputs = []
        list_file_path = None
        out_path = None
        try:
            for i, data in enumerate(valid):
                f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                f.write(data)
                f.flush()
                f.close()
                tmp_inputs.append(f.name)

            # Build concat list file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as lf:
                list_file_path = lf.name
                for p in tmp_inputs:
                    lf.write(f"file '{p}'\n")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as outf:
                out_path = outf.name

            cmd = [
                ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                out_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error("ffmpeg concat failed: %s", proc.stderr.strip())
                if strict:
                    raise RuntimeError("ffmpeg concat failed")
                return merge_wav_pydub(valid)
            with open(out_path, "rb") as f_out:
                merged = f_out.read()
            if len(merged) < 500:
                if strict:
                    raise RuntimeError("Merged WAV unexpectedly small.")
                logger.warning("Merged WAV small; falling back.")
                return merge_wav_pydub(valid)
            return merged
        except Exception as e:
            logger.warning("ffmpeg merge failed (%s). Fallback to pydub.", e)
            if strict:
                raise
            return merge_wav_pydub(valid)
        finally:
            # Cleanup
            if list_file_path and os.path.exists(list_file_path):
                os.remove(list_file_path)
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
            for p in tmp_inputs:
                if os.path.exists(p):
                    os.remove(p)
    else:
        if use_ffmpeg:
            logger.info("ffmpeg not found; using pydub fallback.")
        return merge_wav_pydub(valid)


# +
import os, random, tempfile, logging, time, math, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from itertools import count

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tune these
MAX_WORKERS          = 6          # Kokoro is lightweight; keep modest to avoid rate limits
PER_LINE_RETRIES     = 2          # Additional retries beyond KokoroTTS internal retry
RETRY_SLEEP_BASE_SEC = 1.5
SKIP_EMPTY           = True
MIN_LINE_LEN         = 3          # Ignore super-short fragments that cause “No audio…” failures

def _clean_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        s = (ln or "").strip()
        if SKIP_EMPTY and not s:
            continue
        if len(s) < MIN_LINE_LEN:
            continue
        out.append(s)
    return out

def _backoff_sleep(attempt: int):
    time.sleep(RETRY_SLEEP_BASE_SEC * (attempt + 1))

def _tts_task(task_id: int, text: str, voice: str) -> Tuple[int, bytes | None, str]:
    """
    Returns (task_id, audio_bytes or None, status_msg).
    Resilient: catches ModelError "No audio ..." and returns None (skip).
    """
    t0 = time.time()
    tts = KokoroTTS(voice_id=voice)
    for attempt in range(PER_LINE_RETRIES + 1):
        try:
            audio = tts.tts(text)
            dur = time.time() - t0
            return task_id, audio, f"OK voice={voice} chars={len(text)} time={dur:.2f}s"
        except Exception as e:
            msg = str(e)
            if "No audio was generated" in msg or "Empty/too-small audio" in msg:
                return task_id, None, f"SKIP(no-audio) voice={voice} chars={len(text)}"
            if attempt < PER_LINE_RETRIES:
                _backoff_sleep(attempt)
                continue
            # Final failure
            tb = traceback.format_exc(limit=1).strip().replace("\n", " | ")
            return task_id, None, f"FAIL voice={voice} chars={len(text)} err={tb}"

def run(narticles: int = 1):
    voices = KokoroTTS.list_voices()
    oc_email = os.environ.get("OVERCAST_EMAIL")
    oc_pass  = os.environ.get("OVERCAST_PASSWORD")
    oc_uploader = OvercastUploader(oc_email, oc_pass) if oc_email and oc_pass else None

    sources: List[Tuple[Any, str]] = [(MoneyStuff(), "MoneyStuff")]  # extend if needed

    for src, prefix in sources:
        print(f"\n=== SOURCE {prefix} ===")
        articles = src.list()
        print(f"Found {len(articles)} articles")
        for article_idx, article in enumerate(articles[:narticles], 1):
            title = article.title or "<untitled>"
            print(f"\n[{article_idx}/{len(articles)}] Article: {title}")
            logger.info("Fetching lines...")
            lines_obj = src.get(article)
            raw_lines = getattr(lines_obj, "raw_lines", [])
            lines = _clean_lines(raw_lines)
            if not lines:
                print("No valid lines after cleaning; skipping article.")
                continue
            print(f"Lines: {len(lines)} (after cleaning)")

            # Prepare tasks
            futures = []
            audio_segments: List[bytes] = [b""] * len(lines)
            seq = count()
            submitted = 0

            start_batch = time.time()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {}
                for line_idx, line in enumerate(lines):
                    voice_name = random.choice(voices)
                    fut = ex.submit(_tts_task, line_idx, line, voice_name)
                    futures[fut] = line_idx
            
                submitted = len(futures)
                done_count = 0
                last_log = 0
                audio_segments = [b""] * submitted
            
                for fut in as_completed(futures):
                    line_idx = futures[fut]
                    try:
                        task_id, audio, status = fut.result()
                    except Exception as e:
                        task_id, audio, status = -1, None, f"FAIL (exception) idx={line_idx} err={e}"
                    done_count += 1
                    if audio:
                        audio_segments[line_idx] = audio
                    if done_count - last_log >= 3 or done_count == submitted:
                        pct = 100.0 * done_count / submitted
                        print(f"Progress: {done_count}/{submitted} ({pct:.1f}%)")
                        last_log = done_count
                    print("  ", status)

            # Filter out failed/empty segments
            good_segments = [seg for seg in audio_segments if seg and len(seg) > 500]
            skipped = len(audio_segments) - len(good_segments)
            if not good_segments:
                print("All segments failed or empty; skipping article.")
                continue

            print(f"Merge {len(good_segments)} segments (skipped {skipped}).")
            merged_audio = merge_wav(good_segments)

            wall = time.time() - start_batch
            size_kb = len(merged_audio) / 1024
            print(f"Merged size={size_kb:.1f} KB time={wall:.2f}s avg_per_line={wall/len(lines):.2f}s")

            if oc_uploader:
                try:
                    with tempfile.NamedTemporaryFile(prefix=article.title, suffix=".wav", delete=False) as tmp:
                        tmp.write(merged_audio)
                        tmp.flush()
                        up_start = time.time()
                        oc_uploader.upload(tmp.name)
                        print(f"Uploaded to Overcast in {time.time()-up_start:.2f}s: {title}")
                except Exception as e:
                    logger.exception("Upload failed: %s", e)
            else:
                ep = Episode(topic=title)
                ep.sounds.append(merged_audio)
                ep.texts.extend(lines_obj)
                try:
                    ep.upload(f"[{prefix}] {title[:200]}", f"{prefix} tts: {title}")
                    print("Episode object upload complete.")
                except Exception as e:
                    logger.exception("Episode upload failed: %s", e)

    print("\nRun complete.")
# -



# +
# run(1)

# +
# # !pip install --upgrade beautifulsoup4
# -

