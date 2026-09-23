"""Script fine-tune / training thêm cho Laya trên dữ liệu Tiếng Việt
và tự động đóng gói (export) thành Model mới hoàn chỉnh chuẩn Laya format.

Cách sử dụng:
  python train.py --data data_sample.json --output_dir ./my_laya_vi --epochs 5 --lr 2e-5
"""

import argparse
import json
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoConfig, AutoModel
from laya.common import DecisionModel, build_sequence, collate_items, QTYPES


def load_dataset(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_item(tok, item: dict, max_len: int = 512, head_max_len: int = 192):
    state = item["state"]
    q_dict = item["question"]
    target_label = item["label"]

    qtype_str = q_dict["type"]
    if qtype_str not in QTYPES:
        raise ValueError(f"Loại câu hỏi '{qtype_str}' không hợp lệ. Phải là choice, score hoặc noul.")

    q = {
        "t": qtype_str,
        "ins": q_dict["instructions"],
        "crit": q_dict.get("criteria", {})
    }

    ids, markers = build_sequence(tok, state, q, max_len=max_len, head_max_len=head_max_len)

    # Xác định index nhãn mục tiêu
    crit = q.get("crit", {})
    if qtype_str == "choice":
        options = list(crit.keys())
        if target_label not in options:
            raise ValueError(f"Label '{target_label}' không nằm trong criteria: {options}")
        label_idx = options.index(target_label)
    elif qtype_str == "noul":
        # Noul chuẩn laya: [false, true] -> false=0, true=1
        if isinstance(target_label, bool):
            label_idx = 1 if target_label else 0
        else:
            label_idx = 1 if str(target_label).lower() in ("true", "1", "yes") else 0
    elif qtype_str == "score":
        label_idx = int(target_label)
    else:
        label_idx = int(target_label)

    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[qtype_str],
        "label": label_idx
    }


def train_and_package(
    base_model_dir: str,
    data_path: str,
    output_dir: str,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 2,
    device: str = None
):
    print("=" * 65)
    print(" BẮT ĐẦU TRAINING & ĐÓNG GÓI MODEL LAYA TIẾNG VIỆT")
    print("=" * 65)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[*] Thiết bị tính toán (Device): {device}")

    # 1. Nạp cấu hình và weights từ base checkpoint
    print(f"[*] Đang nạp base model từ: {base_model_dir}...")
    cfg_path = os.path.join(base_model_dir, "rl_agent_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tok_dir = os.path.join(base_model_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir)

    enc_dir = os.path.join(base_model_dir, "encoder")
    ecfg = AutoConfig.from_pretrained(enc_dir)
    enc = AutoModel.from_config(ecfg)

    model = DecisionModel(enc, head_layers=cfg.get("head_layers", 2), n_act=len(cfg.get("act_costs", {})) + 1)
    weights_path = os.path.join(base_model_dir, "model.safetensors")
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=True)
    model.to(device)

    # 2. Xử lý tập dữ liệu
    print(f"[*] Đang tải dữ liệu huấn luyện từ: {data_path}...")
    raw_data = load_dataset(data_path)
    dataset = [prepare_item(tok, d) for d in raw_data]
    print(f"[+] Đã xử lý {len(dataset)} mẫu dữ liệu huấn luyện.")

    # 3. Vòng lặp Fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    print(f"[*] Bắt đầu huấn luyện {epochs} epochs (Batch size: {batch_size}, LR: {lr})...")
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        # Chia batch đơn giản
        for i in range(0, len(dataset), batch_size):
            batch_slice = dataset[i : i + batch_size]
            collated = collate_items([[it] for it in batch_slice], pad_id=pad_id)
            if collated is None:
                continue

            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            marker_pos = collated["marker_pos"].to(device)
            marker_mask = collated["marker_mask"].to(device)
            qtype = collated["qtype"].to(device)
            labels = collated["label"].to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                marker_pos=marker_pos,
                marker_mask=marker_mask,
                qtype=qtype
            )
            logits = out[0]  # [B, K]

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"  [Epoch {epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

    print("[+] Quá trình huấn luyện hoàn tất!")

    # 4. Đóng gói thành Model Laya mới
    print(f"[*] Đang đóng gói model vào thư mục đích: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # 4.1. Lưu weights safetensors
    out_weights_path = os.path.join(output_dir, "model.safetensors")
    cpu_state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(cpu_state_dict, out_weights_path)
    print(f"  [✓] Đã lưu weights: {out_weights_path}")

    # 4.2. Lưu config RL agent
    cfg["model_name"] = os.path.basename(output_dir)
    out_cfg_path = os.path.join(output_dir, "rl_agent_config.json")
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  [✓] Đã lưu config: {out_cfg_path}")

    # 4.3. Copy hoặc lưu Encoder config
    out_enc_dir = os.path.join(output_dir, "encoder")
    os.makedirs(out_enc_dir, exist_ok=True)
    ecfg.save_pretrained(out_enc_dir)
    print(f"  [✓] Đã lưu encoder config: {out_enc_dir}")

    # 4.4. Copy hoặc lưu Tokenizer
    out_tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(out_tok_dir, exist_ok=True)
    tok.save_pretrained(out_tok_dir)
    print(f"  [✓] Đã lưu tokenizer: {out_tok_dir}")

    print("=" * 65)
    print(f" ĐÓNG GÓI MODEL THÀNH CÔNG: {output_dir}")
    print("=" * 65)
    print(f"Bạn có thể nạp ngay model mới bằng lệnh:")
    print(f"  agent = Agent('{output_dir}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune và đóng gói Model Laya")
    parser.add_argument("--data", type=str, default="data_sample.json", help="Đường dẫn file JSON dữ liệu huấn luyện")
    parser.add_argument("--output_dir", type=str, default="./my_laya_vietnamese", help="Thư mục xuất model mới")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện")
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học (Learning rate)")
    parser.add_argument("--batch_size", type=int, default=2, help="Kích thước batch")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/Users/duc/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual",
        help="Đường dẫn base checkpoint multilingual"
    )
    args = parser.parse_args()

    train_and_package(
        base_model_dir=args.base_model_dir,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
