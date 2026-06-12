import html
import re
import unicodedata


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REDDIT_USER_RE = re.compile(r"/?u/[A-Za-z0-9_\-]+")
_REDDIT_SUB_RE = re.compile(r"/?r/[A-Za-z0-9_\-]+")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)\*")
_MD_QUOTE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")

# Minimal embedded contractions map (avoid adding `contractions` dependency)
_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "can't": "cannot",
    "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "it's": "it is", "i'm": "i am", "you're": "you are", "they're": "they are",
    "we're": "we are", "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he's": "he is", "she's": "she is", "that's": "that is", "what's": "what is",
}


def clean_text(text: str, *, lowercase: bool = True) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _REDDIT_USER_RE.sub(" ", text)
    text = _REDDIT_SUB_RE.sub(" ", text)
    text = _MD_QUOTE_RE.sub(" ", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    if lowercase:
        text = text.lower()
    for short, long in _CONTRACTIONS.items():
        text = text.replace(short, long)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
