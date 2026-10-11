"""Automatic answer checkers for the routing evaluation set.

Every item has a `check` spec. `score(response, check)` returns True/False.
Code answers run in a separate, isolated Python process (see `run_python_tests`).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

WORD_RE = re.compile(r"[A-Za-z0-9']+")
NUM_RE = r"-?\$?\d[\d,]*(?:\.\d+)?|-?\.\d+"


# --------------------------------------------------------------------------- helpers

def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_number(text: str) -> Optional[float]:
    """The number after the last 'Answer:' marker, else the last number in the text."""
    text = strip_think(text)
    marked = re.findall(r"answer\s*(?:is)?\s*[:=\-]?\s*\**\s*(" + NUM_RE + ")", text, flags=re.I)
    if marked:
        return _to_float(marked[-1])
    nums = re.findall(NUM_RE, text)
    return _to_float(nums[-1]) if nums else None


def extract_label(text: str, options: List[str]) -> Optional[str]:
    """The option after the last 'Answer:' marker, else the last option mentioned."""
    text = strip_think(text)
    alt = "|".join(re.escape(o) for o in sorted(options, key=len, reverse=True))
    marked = re.findall(r"answer\s*(?:is)?\s*[:=\-]?\s*[\*\"'(\[]*\s*(" + alt + r")\b", text, flags=re.I)
    if marked:
        return marked[-1].upper()
    found = [(m.start(), m.group(1)) for m in re.finditer(r"(?<![A-Za-z0-9_])(" + alt + r")(?![A-Za-z0-9_])", text, flags=re.I)]
    return found[-1][1].upper() if found else None


def extract_choice(text: str, options=("A", "B", "C", "D")) -> Optional[str]:
    """Multiple choice letter: 'Answer: B', '(B)', 'B.' or a lone letter."""
    text = strip_think(text)
    letters = "".join(options)
    pats = [r"answer\s*(?:is)?\s*[:=\-]?\s*[\*\(\[]*\s*([" + letters + r"])\b",
            r"\(([" + letters + r"])\)", r"^\s*([" + letters + r"])[\.\):]", r"\b([" + letters + r"])\b"]
    for p in pats:
        m = re.findall(p, text, flags=re.M | (re.I if p.startswith("answer") else 0))
        if m:
            return m[-1].upper()
    return None


def content_lines(text: str) -> List[str]:
    """Non-empty lines, dropping a lead-in line that ends with ':' (e.g. 'Here is a poem:')
    and surrounding quotes or markdown emphasis."""
    lines = [l.strip().strip('"*').strip() for l in strip_think(text).splitlines()]
    lines = [l for l in lines if l]
    if len(lines) > 1 and lines[0].endswith(":"):
        lines = lines[1:]
    return lines


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _contains(text: str, term: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])", text.lower()) is not None


def extract_code(text: str, entry: Optional[str] = None) -> str:
    """Python code from a fenced block (preferring one that defines `entry`), else the raw text."""
    text = strip_think(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if blocks:
        if entry:
            for b in blocks:
                if re.search(r"def\s+" + re.escape(entry) + r"\s*\(", b):
                    return b
        return max(blocks, key=len)
    m = re.search(r"^(?:from|import|def|class)\b", text, flags=re.M)
    return text[m.start():] if m else text


# --------------------------------------------------------------------------- sandbox

_RUNNER = r'''
import sys, os, json
spec = json.loads(sys.stdin.read())
BLOCK = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate",
         "os.symlink", "os.link", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
         "os.startfile", "os.kill", "os.putenv", "os.unsetenv", "shutil.rmtree", "shutil.copyfile",
         "shutil.move", "shutil.copytree", "subprocess.Popen", "_winapi.CreateProcess",
         "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto",
         "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function", "builtins.input",
         "webbrowser.open", "urllib.Request", "http.client.connect"}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
def hook(event, args):
    if event in BLOCK or event.startswith("winreg.") and not event.startswith(("winreg.OpenKey", "winreg.QueryValue", "winreg.EnumKey", "winreg.EnumValue", "winreg.ConnectRegistry", "winreg.PyHKEY")):
        raise PermissionError("blocked in sandbox: " + event)
    if event == "open":
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS):
            raise PermissionError("blocked in sandbox: file write")
sys.addaudithook(hook)
ns = {"__name__": "candidate"}
try:
    exec(spec["setup"], ns)
    exec(spec["code"], ns)
    for t in spec["tests"]:
        exec(t, ns)
except BaseException as e:
    print("FAIL", type(e).__name__, str(e)[:200])
    sys.exit(1)
print("PASS")
'''


def run_python_tests(code: str, tests: List[str], setup: str = "", timeout: float = 10.0) -> bool:
    """Run untrusted model code plus asserts in an isolated child interpreter.

    Isolation: `python -I` (ignores env vars and user site-packages), an empty
    temp working directory, a timeout, and an audit hook that blocks file writes,
    process creation, network access, ctypes and input().
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", _RUNNER],
                                  input=json.dumps({"code": code, "tests": tests, "setup": setup}),
                                  capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0 and proc.stdout.strip().endswith("PASS")


# --------------------------------------------------------------------------- scoring

def check_constraints(text: str, c: Dict[str, Any]) -> bool:
    lines = content_lines(text)
    body = "\n".join(lines)
    w = words(body)
    if "lines" in c and len(lines) != c["lines"]:
        return False
    if "max_words" in c and len(w) > c["max_words"]:
        return False
    if "min_words" in c and len(w) < c["min_words"]:
        return False
    if "words_exact" in c and len(w) != c["words_exact"]:
        return False
    for term in c.get("include", []):
        if not _contains(body, term):
            return False
    for alts in c.get("include_any", []):
        if not any(_contains(body, t) for t in alts):
            return False
    for term in c.get("exclude", []):
        if _contains(body, term):
            return False
    for ch in c.get("exclude_chars", []):
        if ch.lower() in body.lower():
            return False
    if "max_words_per_line" in c and any(len(words(l)) > c["max_words_per_line"] for l in lines):
        return False
    if "lines_start" in c:
        starts = c["lines_start"]
        if len(lines) < len(starts):
            return False
        for line, s in zip(lines, starts):
            if s is not None and not line.lower().startswith(s.lower()):
                return False
    for idx, terms in c.get("line_contains", {}).items():
        i = int(idx)
        if not lines or abs(i) > len(lines) - (0 if i < 0 else 1):
            return False
        if not all(_contains(lines[i], t) for t in terms):
            return False
    if "same_lines" in c:
        a, b = c["same_lines"]
        if len(lines) <= max(a, b) or lines[a].rstrip(".,!;").lower() != lines[b].rstrip(".,!;").lower():
            return False
    if "numbered_items" in c and len([l for l in lines if re.match(r"^\d+[\.\)]", l)]) != c["numbered_items"]:
        return False
    if "question_marks" in c and body.count("?") != c["question_marks"]:
        return False
    if c.get("lowercase") and any(ch.isupper() for ch in body):
        return False
    if "ends_with" in c and not re.sub(r"[^a-z0-9]+$", "", body.lower()).endswith(c["ends_with"].lower()):
        return False
    if "equals" in c and re.sub(r"[^a-z0-9 ]", "", body.lower()).strip() != c["equals"].lower():
        return False
    return True


def _parse_json(text: str):
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except ValueError:
            continue
    return None


def _norm(v):
    if isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v.strip()):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k.lower(): _norm(x) for k, x in v.items()}
    return v


def score(response: str, check: Dict[str, Any]) -> bool:
    t = check["type"]
    if t == "number":
        got = extract_number(response)
        tol = check.get("tol", 1e-6)
        return got is not None and abs(got - check["answer"]) <= tol * max(1.0, abs(check["answer"]))
    if t == "choice":
        return extract_choice(response, tuple(check.get("options", "ABCD"))) == check["answer"]
    if t == "label":
        return extract_label(response, check["options"]) == check["answer"].upper()
    if t == "label_sequence":
        alt = "|".join(re.escape(o) for o in check["options"])
        found = [m.upper() for m in re.findall(r"(?<![A-Za-z])(" + alt + r")(?![A-Za-z])", strip_think(response), flags=re.I)]
        return found == [a.upper() for a in check["answer"]]
    if t == "python":
        return run_python_tests(extract_code(response, check.get("entry")), check["tests"], check.get("setup", ""))
    if t == "constraints":
        return check_constraints(response, check)
    if t == "ordered":
        body = strip_think(response).lower()
        pos = [body.find(x.lower()) for x in check["items"]]
        return all(p >= 0 for p in pos) and pos == sorted(pos)
    if t == "json_keys":
        obj = _parse_json(response)
        return isinstance(obj, dict) and set(k.lower() for k in obj) >= set(k.lower() for k in check["keys"])
    if t == "json_equals":
        return _norm(_parse_json(response)) == _norm(check["answer"])
    raise ValueError(f"unknown check type {t}")
