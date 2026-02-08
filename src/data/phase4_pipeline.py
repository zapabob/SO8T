"""Phase4 data pipeline: API fetch, PDF ingest, HF CLI integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _resolve_json_path(data: Any, path: str) -> Any:
    cur = data
    for part in path.split('.'):
        if part == '':
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return None
    return cur


def _request_with_backoff(
    url: str,
    headers: Dict[str, Any],
    params: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff_seconds: float,
    min_interval_seconds: float,
    last_request_time: Optional[float],
) -> Tuple[requests.Response, float]:
    attempt = 0
    while True:
        if last_request_time is not None:
            elapsed = time.time() - last_request_time
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        last_request_time = time.time()
        if resp.status_code in {429, 500, 502, 503, 504} and attempt < retries:
            time.sleep(backoff_seconds * (2 ** attempt))
            attempt += 1
            continue
        resp.raise_for_status()
        return resp, last_request_time


def _paginate_requests(
    src: Dict[str, Any],
    headers: Dict[str, Any],
    params: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff_seconds: float,
    min_interval_seconds: float,
) -> Iterable[Dict[str, Any]]:
    pagination = src.get('pagination') or {}
    mode = pagination.get('type')
    max_pages = pagination.get('max_pages', 1)

    last_request_time = None
    if not mode:
        resp, last_request_time = _request_with_backoff(
            src['url'], headers, params, timeout, retries, backoff_seconds, min_interval_seconds, last_request_time
        )
        yield resp.json() if 'application/json' in resp.headers.get('content-type', '') else resp.text
        return

    if mode == 'page':
        param = pagination.get('param', 'page')
        start = pagination.get('start', 1)
        size_param = pagination.get('page_size_param')
        size = pagination.get('page_size')
        for page in range(start, start + max_pages):
            page_params = dict(params)
            page_params[param] = page
            if size_param and size is not None:
                page_params[size_param] = size
            resp, last_request_time = _request_with_backoff(
                src['url'], headers, page_params, timeout, retries, backoff_seconds, min_interval_seconds, last_request_time
            )
            yield resp.json() if 'application/json' in resp.headers.get('content-type', '') else resp.text
        return

    if mode == 'offset':
        param = pagination.get('param', 'offset')
        size_param = pagination.get('page_size_param', 'limit')
        size = pagination.get('page_size', 100)
        offset = pagination.get('start', 0)
        for _ in range(max_pages):
            page_params = dict(params)
            page_params[param] = offset
            page_params[size_param] = size
            resp, last_request_time = _request_with_backoff(
                src['url'], headers, page_params, timeout, retries, backoff_seconds, min_interval_seconds, last_request_time
            )
            yield resp.json() if 'application/json' in resp.headers.get('content-type', '') else resp.text
            offset += size
        return

    if mode == 'cursor':
        cursor_param = pagination.get('param', 'cursor')
        cursor_path = pagination.get('cursor_path')
        cursor = pagination.get('start')
        for _ in range(max_pages):
            page_params = dict(params)
            if cursor is not None:
                page_params[cursor_param] = cursor
            resp, last_request_time = _request_with_backoff(
                src['url'], headers, page_params, timeout, retries, backoff_seconds, min_interval_seconds, last_request_time
            )
            data = resp.json() if 'application/json' in resp.headers.get('content-type', '') else resp.text
            yield data
            if cursor_path:
                cursor = _resolve_json_path(data, cursor_path)
                if not cursor:
                    break
        return


def fetch_api_sources(api_sources: List[Dict[str, Any]], out_dir: Path, defaults: Dict[str, Any]) -> List[Path]:
    _ensure_dir(out_dir)
    saved = []
    for src in api_sources:
        name = src.get('name', 'source')
        headers = src.get('headers') or {}
        params = src.get('params') or {}
        timeout = src.get('timeout', defaults.get('timeout', 30))
        retries = src.get('retries', defaults.get('retries', 3))
        backoff_seconds = src.get('backoff_seconds', defaults.get('backoff_seconds', 1.0))
        min_interval_seconds = src.get('min_interval_seconds', defaults.get('min_interval_seconds', 0.0))
        results_path = src.get('results_path')

        pages = list(_paginate_requests(src, headers, params, timeout, retries, backoff_seconds, min_interval_seconds))
        if results_path:
            items = []
            for page in pages:
                extracted = _resolve_json_path(page, results_path) if isinstance(page, dict) else None
                if isinstance(extracted, list):
                    items.extend(extracted)
            data = {'meta': {'pages': len(pages)}, 'items': items}
        else:
            data = {'meta': {'pages': len(pages)}, 'pages': pages}

        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

        expected = src.get('sha256')
        if expected:
            actual = _sha256_path(out_path)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {out_path}: {actual} != {expected}")
        saved.append(out_path)
    return saved


def _download_file(url: str, out_path: Path, timeout: int, min_interval_seconds: float) -> Path:
    time.sleep(min_interval_seconds) if min_interval_seconds else None
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with out_path.open('wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path


def ingest_pdfs(pdf_sources: List[Dict[str, Any]], out_dir: Path, defaults: Dict[str, Any]) -> List[Path]:
    if PdfReader is None:
        raise RuntimeError('pypdf is required for PDF ingest. Install pypdf>=4.0.0')
    _ensure_dir(out_dir)
    outputs = []
    for src in pdf_sources:
        name = src.get('name', 'document')
        url = src.get('url')
        local = src.get('path')
        timeout = src.get('timeout', defaults.get('timeout', 60))
        min_interval_seconds = src.get('min_interval_seconds', defaults.get('min_interval_seconds', 0.0))
        pdf_path = None
        if url:
            pdf_path = out_dir / f"{name}.pdf"
            _download_file(url, pdf_path, timeout=timeout, min_interval_seconds=min_interval_seconds)
        elif local:
            pdf_path = Path(local)
        else:
            continue

        expected = src.get('sha256')
        if expected:
            actual = _sha256_path(pdf_path)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {pdf_path}: {actual} != {expected}")

        reader = PdfReader(str(pdf_path))
        jsonl_path = out_dir / f"{name}.jsonl"
        with jsonl_path.open('w', encoding='utf-8') as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                record = {
                    'source': name,
                    'page': i + 1,
                    'text': text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        outputs.append(jsonl_path)
    return outputs


def hf_cli_download(hf_sources: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    _ensure_dir(out_dir)
    outputs = []
    for src in hf_sources:
        repo = src['repo']
        repo_type = src.get('type', 'dataset')
        allow = src.get('allow')
        revision = src.get('revision')
        cmd = ['huggingface-cli', 'download', repo, '--repo-type', repo_type, '--local-dir', str(out_dir / repo.replace('/', '__'))]
        if allow:
            for pattern in allow:
                cmd.extend(['--allow', pattern])
        if revision:
            cmd.extend(['--revision', revision])
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"HF CLI download failed for {repo}: {result.stderr}")
        repo_dir = out_dir / repo.replace('/', '__')
        checksums = src.get('checksums') or []
        for item in checksums:
            rel_path = item.get('path')
            expected = item.get('sha256')
            if not rel_path or not expected:
                continue
            target = repo_dir / rel_path
            if not target.exists():
                raise FileNotFoundError(target)
            actual = _sha256_path(target)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {target}: {actual} != {expected}")
        outputs.append(repo_dir)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase4 data pipeline: API fetch, PDF ingest, HF CLI integration')
    parser.add_argument('--config', default='config/phase4_pipeline.yaml')
    parser.add_argument('--out', default='data/phase4')
    parser.add_argument('--skip-api', action='store_true')
    parser.add_argument('--skip-pdf', action='store_true')
    parser.add_argument('--skip-hf', action='store_true')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = Path(args.out)
    defaults = cfg.get('defaults', {})

    if not args.skip_api:
        fetch_api_sources(cfg.get('api_sources', []), out_root / 'api', defaults)
    if not args.skip_pdf:
        ingest_pdfs(cfg.get('pdf_sources', []), out_root / 'pdf', defaults)
    if not args.skip_hf:
        hf_cli_download(cfg.get('hf_sources', []), out_root / 'hf')


if __name__ == '__main__':
    main()
