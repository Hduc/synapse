"""Demo sử dụng Executor: Chạy lệnh CLI hoặc Gọi API với Auth & Dynamic Headers."""

import time
import hashlib
import hmac
from executor import execute_cli, call_api, dispatch_command


def test_cli_execution():
    print("\n" + "=" * 60)
    print("1. TEST CHẠY LỆNH CLI (HỆ THỐNG)")
    print("=" * 60)

    # Ví dụ 1: Lệnh kiểm tra RAM
    action_ram = {
        "type": "cli",
        "command": "top -l 1 -s 0 | grep -E 'PhysMem'",
        "timeout": 10
    }
    res_ram = dispatch_command(action_ram)
    print(f"Lệnh: {action_ram['command']}")
    print(f"Trạng thái: {res_ram['status']}")
    print(f"Kết quả (Output): {res_ram['output']}")
    print(f"Thời gian chạy: {res_ram['elapsed_ms']} ms")

    # Ví dụ 2: Lệnh CLI có truyền tham số động ({target_path})
    action_ls = {
        "type": "cli",
        "command": "ls -lh {target_path} | head -n 4",
        "timeout": 5
    }
    context = {"target_path": "/Users/duc/.openclaw/workspace/synapse"}
    res_ls = dispatch_command(action_ls, context=context)
    print(f"\nLệnh (đã nội suy): {res_ls['command']}")
    print(f"Kết quả:\n{res_ls['output']}")


def test_api_calls():
    print("\n" + "=" * 60)
    print("2. TEST GỌI REST API (AUTH & DYNAMIC HEADERS)")
    print("=" * 60)

    # Hàm sinh Dynamic Header (ví dụ: tạo timestamp, request-id, và chữ ký HMAC)
    def my_dynamic_header_generator(ctx):
        current_time = str(int(time.time()))
        secret_key = "my-secret-key"
        # Tạo chữ ký HMAC SHA256 động cho mỗi request
        signature = hmac.new(
            secret_key.encode(),
            current_time.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "X-Timestamp": current_time,
            "X-Signature": signature,
            "X-Request-ID": f"req-{ctx.get('user_id', 'anonymous')}-{current_time}"
        }

    # Cấu hình Action gọi API
    action_api = {
        "type": "api",
        "url": "https://httpbin.org/headers",
        "method": "GET",
        "headers": {
            "User-Agent": "Laya-Agent/1.0",
            "Accept": "application/json"
        },
        # Header động tạo bằng hàm callable:
        "dynamic_headers": my_dynamic_header_generator,
        # Cấu hình Xác thực (Bearer Token, API-Key, hoặc Basic):
        "auth": {
            "type": "bearer",
            "token": "token-xyz-987654"
        },
        "timeout": 15.0
    }

    context = {"user_id": "duc_hoang"}
    res_api = dispatch_command(action_api, context=context)

    print(f"URL: {res_api['url']}")
    print(f"HTTP Status: {res_api.get('status_code')}")
    print(f"Thời gian gọi: {res_api['elapsed_ms']} ms")
    print("Các headers phía Server nhận được:")
    headers_received = res_api["data"].get("headers", {})
    for k in ["Authorization", "User-Agent", "X-Timestamp", "X-Signature", "X-Request-Id"]:
        if k in headers_received:
            print(f"  • {k}: {headers_received[k]}")


if __name__ == "__main__":
    test_cli_execution()
    test_api_calls()
