"""Script chạy inference Laya cho bài toán Tiếng Việt.

Sử dụng:
  python infer.py --model convaiinnovations/laya --subfolder multilingual
  python infer.py --model ./my_laya_model
"""

import argparse
import json
import sys
from laya import Agent


def run_inference(model_path: str, subfolder: str = None):
    print(f"[*] Đang nạp model từ: {model_path} (subfolder={subfolder})...")
    agent = Agent(model_path, subfolder=subfolder)
    print(f"[+] Model đã sẵn sàng trên thiết bị: {agent.device}")

    # Test Case 1: Phân loại ý định khách hàng (Type: choice)
    state_ticket = "Tôi đã chuyển khoản 500k qua Vietcombank từ sáng nhưng tài khoản web vẫn chưa lên gói Pro. Hỗ trợ kích hoạt gấp giúp tôi!"
    q_intent = {
        "intent": {
            "type": "choice",
            "instructions": "Phân loại ý định của khách hàng trong tin nhắn hỗ trợ:",
            "criteria": {
                "payment_issue": "Thanh toán chưa nhận được tiền hoặc chưa kích hoạt gói",
                "technical_help": "Lỗi phần mềm hoặc giao diện không hoạt động",
                "cancellation": "Yêu cầu hủy tài khoản hoặc hoàn phí",
                "general_inquiry": "Hỏi thông tin chung về tính năng và bảng giá"
            }
        }
    }

    res_intent = agent.predict(state_ticket, q_intent)
    ans1 = res_intent["answers"]["intent"]

    print("\n" + "=" * 65)
    print("VÍ DỤ 1: PHÂN LOẠI Ý ĐỊNH KHÁCH HÀNG (Type: Choice)")
    print("=" * 65)
    print(f"Nội dung đầu vào: \"{state_ticket}\"")
    print(f"-> Lựa chọn dự đoán: {ans1['choice']}")
    print(f"-> Độ tin cậy (Confidence): {ans1['confidence']:.2%}")
    print(f"-> Xác suất chi tiết:")
    for opt, prob in ans1["probabilities"].items():
        print(f"   • {opt}: {prob:.4f}")
    if "action" in ans1:
        print(f"-> Quyết định hành động: {ans1['action']}")

    # Test Case 2: Kiểm tra khiếu nại bức xúc / tiêu cực (Type: noul - Boolean true/false)
    state_comment = "Dịch vụ bên bạn làm ăn quá tắc trách, nhắn tin nửa ngày không ai trả lời, tôi rất bực mình!"
    q_sentiment = {
        "is_negative_complaint": {
            "type": "noul",
            "instructions": "Tin nhắn này có chứa thái độ tiêu cực, bức xúc hoặc khiếu nại gay gắt không?",
            "criteria": {
                "false": "Không, tin nhắn bình thường, trung tính hoặc hỏi thăm tích cực",
                "true": "Có, khách hàng đang bức xúc, phàn nàn gay gắt hoặc khiếu nại"
            }
        }
    }

    res_sentiment = agent.predict(state_comment, q_sentiment)
    ans2 = res_sentiment["answers"]["is_negative_complaint"]

    print("\n" + "=" * 65)
    print("VÍ DỤ 2: PHÁT HIỆN KHIẾU NẠI / TIÊU CỰC (Type: Noul - True/False)")
    print("=" * 65)
    print(f"Nội dung đầu vào: \"{state_comment}\"")
    if "choice" in ans2:
        print(f"-> Kết quả lựa chọn: {ans2['choice']}")
    elif "noul" in ans2:
        val = ans2["noul"]
        is_true = val >= 0.5
        print(f"-> Điểm noul: {val:.4f} => Dự đoán: {'TRUE (Có)' if is_true else 'FALSE (Không)'}")
    print(f"-> Độ tin cậy (Confidence): {ans2['confidence']:.2%}")
    if "probabilities" in ans2:
        print(f"-> Xác suất chi tiết:")
        for opt, prob in ans2["probabilities"].items():
            print(f"   • {opt}: {prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laya Vietnamese Inference")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="Hugging Face model ID hoặc đường dẫn thư mục model")
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder nếu repo chứa nhiều biến thể (vd: multilingual). Mặc định tự động nhận diện.")
    args = parser.parse_args()

    sub = args.subfolder
    if sub is None and args.model == "convaiinnovations/laya":
        sub = "multilingual"
    elif sub and sub.lower() in ("none", "", "null"):
        sub = None

    run_inference(args.model, subfolder=sub)
