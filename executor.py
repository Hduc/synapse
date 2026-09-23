"""Bộ công cụ thực thi lệnh CLI và gọi API (Executor & Dispatcher).

Tính năng:
1. execute_cli: Chạy lệnh shell (CLI), bắt stdout, stderr, exit_code, timeout.
2. call_api: Gọi HTTP API với:
   - Auth đa dạng (Bearer token, API Key header/param, Basic auth).
   - Dynamic Headers (Header động):
     + Dict nội suy template: {"Authorization": "Bearer {token}", "X-Date": "{date}"}
     + Hàm callable tạo header theo thời gian thực (HMAC, Timestamp, Nonce, v.v.).
3. dispatch_command: Hàm nhận dạng tổng quát nhận spec hoặc câu lệnh để thực thi tự động.
"""

import os
import subprocess
import time
import hmac
import hashlib
import inspect
from typing import Any, Callable, Dict, Optional, Union
import httpx


def resolve_dynamic_headers(
    headers: Optional[Dict[str, str]] = None,
    dynamic_headers: Optional[Union[Dict[str, Any], Callable[..., Dict[str, str]]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Tổng hợp headers cố định và headers động (hàm callback hoặc nội suy template)."""
    final_headers = {}
    if headers:
        final_headers.update({str(k): str(v) for k, v in headers.items()})

    # 1. Headers tạo động từ Callable (hàm tạo token, sign HMAC, timestamp...)
    if dynamic_headers:
        if callable(dynamic_headers):
            try:
                sig = inspect.signature(dynamic_headers)
                if len(sig.parameters) > 0 and context is not None:
                    generated = dynamic_headers(context)
                else:
                    generated = dynamic_headers()
                if isinstance(generated, dict):
                    final_headers.update({str(k): str(v) for k, v in generated.items()})
            except Exception as e:
                print(f"[Warning] Lỗi khi tạo dynamic_headers từ callable: {e}")
        elif isinstance(dynamic_headers, dict):
            for k, v in dynamic_headers.items():
                if isinstance(v, str) and context and "{" in v and "}" in v:
                    try:
                        v = v.format(**context)
                    except Exception:
                        pass
                final_headers[str(k)] = str(v)

    # 2. Nội suy các biến {variable} trong headers nếu có context
    if context:
        for k, v in list(final_headers.items()):
            if isinstance(v, str) and "{" in v and "}" in v:
                try:
                    final_headers[k] = v.format(**context)
                except Exception:
                    pass

    return final_headers


def execute_cli(
    command: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Thực thi lệnh shell/CLI.

    Ví dụ:
      execute_cli("top -l 1 -s 0 | grep -E 'PhysMem'")
      execute_cli("sudo powermetrics --samplers smc -n 1 | grep -i 'CPU die temperature'")
    """
    if context and ("{" in command and "}" in command):
        try:
            command = command.format(**context)
        except Exception:
            pass

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start_time = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged_env
        )
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "type": "cli",
            "status": "success" if proc.returncode == 0 else "error",
            "command": command,
            "exit_code": proc.returncode,
            "output": proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip(),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "elapsed_ms": elapsed_ms
        }
    except subprocess.TimeoutExpired:
        return {
            "type": "cli",
            "status": "timeout",
            "command": command,
            "exit_code": -1,
            "output": f"Lệnh bị quá thời gian cho phép ({timeout}s)",
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "elapsed_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        return {
            "type": "cli",
            "status": "exception",
            "command": command,
            "exit_code": -1,
            "output": str(e),
            "stdout": "",
            "stderr": str(e),
            "elapsed_ms": round((time.time() - start_time) * 1000, 2)
        }


def call_api(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    dynamic_headers: Optional[Union[Dict[str, Any], Callable[..., Dict[str, str]]]] = None,
    auth: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Gọi REST API hỗ trợ Auth và Dynamic Headers.

    Auth types:
      - Bearer: {"type": "bearer", "token": "..."}
      - API Key Header: {"type": "api_key", "header_name": "X-API-Key", "key": "..."}
      - Basic Auth: {"type": "basic", "username": "...", "password": "..."}
    """
    # 1. Nội suy URL nếu có biến template
    if context and ("{" in url and "}" in url):
        try:
            url = url.format(**context)
        except Exception:
            pass

    # 2. Xử lý dynamic headers
    resolved_headers = resolve_dynamic_headers(headers, dynamic_headers, context)

    # 3. Xử lý Auth
    httpx_auth = None
    if auth:
        auth_type = auth.get("type", "").lower()
        if auth_type == "bearer":
            token = auth.get("token", "")
            if context and isinstance(token, str) and "{" in token:
                try:
                    token = token.format(**context)
                except Exception:
                    pass
            resolved_headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key":
            header_name = auth.get("header_name", "X-API-Key")
            key_val = auth.get("key", "")
            if context and isinstance(key_val, str) and "{" in key_val:
                try:
                    key_val = key_val.format(**context)
                except Exception:
                    pass
            resolved_headers[header_name] = key_val

        elif auth_type == "basic":
            httpx_auth = (auth.get("username", ""), auth.get("password", ""))

    # 4. Gửi HTTP Request
    start_time = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(
                method=method.upper(),
                url=url,
                headers=resolved_headers,
                params=params,
                json=json_body,
                data=data,
                auth=httpx_auth
            )

            try:
                resp_json = resp.json()
            except Exception:
                resp_json = None

            return {
                "type": "api",
                "status": "success" if resp.is_success else "http_error",
                "status_code": resp.status_code,
                "url": str(resp.url),
                "data": resp_json if resp_json is not None else resp.text,
                "headers": dict(resp.headers),
                "elapsed_ms": round((time.time() - start_time) * 1000, 2)
            }
    except httpx.TimeoutException:
        return {
            "type": "api",
            "status": "timeout",
            "url": url,
            "error": f"Request timeout sau {timeout}s",
            "elapsed_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        return {
            "type": "api",
            "status": "exception",
            "url": url,
            "error": str(e),
            "elapsed_ms": round((time.time() - start_time) * 1000, 2)
        }


def dispatch_command(
    action_spec: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Hàm tổng quát nhận diện và thực thi lệnh CLI hoặc gọi API."""
    action_type = action_spec.get("type", "").lower()

    if action_type == "cli":
        return execute_cli(
            command=action_spec["command"],
            context=context,
            timeout=action_spec.get("timeout", 30),
            cwd=action_spec.get("cwd"),
            env=action_spec.get("env")
        )

    elif action_type == "api":
        return call_api(
            url=action_spec["url"],
            method=action_spec.get("method", "GET"),
            headers=action_spec.get("headers"),
            dynamic_headers=action_spec.get("dynamic_headers"),
            auth=action_spec.get("auth"),
            params=action_spec.get("params"),
            json_body=action_spec.get("json_body"),
            data=action_spec.get("data"),
            context=context,
            timeout=action_spec.get("timeout", 30.0)
        )

    else:
        raise ValueError(f"Loại action không hỗ trợ: '{action_type}'. Vui lòng chỉ định 'cli' hoặc 'api'.")
