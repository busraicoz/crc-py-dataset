from __future__ import annotations
import os
import json
from time import sleep
from typing import Dict, List, Tuple
import requests
import html
from dotenv import load_dotenv

from src.crc_py.utils.schema import validate_record

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": "token {GITHUB_TOKEN}" if GITHUB_TOKEN else None,
    "Accept": "application/vnd.github.v3+json"
}
HEADERS = {k: v for k, v in HEADERS.items() if v}

def _normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    s = s.replace("", "").replace("", "")
    s = "".join(line.rstrip() for line in s.split(""))
    return s.strip()

def _make_key(code: str, comment: str) -> Tuple[str, str]:
    return _normalize_text(code), _normalize_text(comment)

def _get_with_pagination(url: str, params: Dict | None = None) -> List[Dict]:
    out: List[Dict] = []
    page = 1
    while True:
        p = dict(params or {})
        p.update({"per_page": 100, "page": page})
        r = requests.get(url, headers=HEADERS, params=p)
        if r.status_code == 403 and 'X-RateLimit-Remaining' in r.headers and r.headers['X-RateLimit-Remaining'] == '0':
            reset = int(r.headers.get('X-RateLimit-Reset', '0'))
            wait = max(reset - int(__import__('time').time()), 30)
            sleep(wait)
            continue
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        page += 1
        if page > 50:
            break
    return out

def get_pull_requests(owner: str, repo: str) -> List[Dict]:
    url = "https://api.github.com/repos/{owner}/{repo}/pulls"
    return _get_with_pagination(url, params={"state": "closed"})

def get_review_comments(owner: str, repo: str, pr_number: int) -> List[Dict]:
    url = "https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    return _get_with_pagination(url)

def crawl(owner: str, repo: str, *, base_path: str | None = None, out_path: str = "out.json") -> None:
    base_seen = set()
    if base_path and os.path.isfile(base_path):
        arr = json.load(open(base_path, 'r', encoding='utf-8'))
        if isinstance(arr, list):
            for obj in arr:
                base_seen.add(_make_key(obj.get('code',''), obj.get('comment','')))
    out_seen = set()
    if os.path.isfile(out_path):
        arr = json.load(open(out_path, 'r', encoding='utf-8'))
        if isinstance(arr, list):
            for obj in arr:
                out_seen.add(_make_key(obj.get('code',''), obj.get('comment','')))
    seen = base_seen | out_seen

    pulls = get_pull_requests(owner, repo)
    data: List[Dict] = []
    added = 0
    for pr in pulls:
        prn = pr.get('number')
        comments = get_review_comments(owner, repo, prn)
        for c in comments:
            code = c.get('diff_hunk', '')
            comment = c.get('body', '')
            if not code or not comment:
                continue
            key = _make_key(code, comment)
            if key in seen:
                continue
            path = c.get('path', '')
            line_no = c.get('original_line') or c.get('line') or c.get('original_start_line') or c.get('start_line')
            enriched = f"File: {path}"
            Code: {code}
            Comment: {comment}
            item = {
                "code": code,
                "comment": comment,
                "line_number": line_no,
                "enriched": enriched,
                "subcategory": "",  # to be filled by classifier or human
                "category": "",     # to be mapped from subcategory
                "file_path": path,
                "pr_number": prn,
                "repo": repo,
                "owner": owner,
                "comment_id": c.get('id'),
                "comment_created_at": c.get('created_at')
            }
            try:
                # We validate only core fields; sub/category may be blank now
                validate_record({k:v for k,v in item.items() if k != 'category' and k != 'subcategory'})
            except Exception:
                pass
            data.append(item)
            seen.add(key)
            added += 1
    merged = data
    if os.path.isfile(out_path):
        existing = json.load(open(out_path, 'r', encoding='utf-8'))
        if isinstance(existing, list):
            merged = existing + data
    json.dump(merged, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print("[ok] {owner}/{repo}: added {added} new items → {out_path}")
