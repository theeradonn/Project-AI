"""
SafeTrade — Phase 4: DPO fine-tuning (รันบน H100)

เทรน LoRA adapter ด้วย DPO จากคู่ที่ build_dpo_pairs.py สร้างไว้
ออกแบบให้รันผ่าน Jupyter ของ LiCO ที่ไม่มี terminal และ session มีเวลาจำกัด
จึงเซฟ checkpoint ถี่และ resume ได้เองโดยไม่ต้องใส่ flag

ประเด็นที่พลาดง่ายและสคริปต์นี้กันไว้แล้ว:

1. DPOConfig ตั้ง max_prompt_length=512 / max_length=1024 เป็นค่า default แต่ prompt จริง
   ยาว ~6,000 token และ TRL จะ "ตัดทิ้งเงียบๆ ไม่ใช่ข้าม" ซ้ำร้าย truncation_mode default
   คือ keep_end ซึ่งตัดหัว catalogue ทิ้ง = บั๊กเดียวกับที่ Ollama ตัด prompt เป๊ะ
   → วัดความยาวจริงก่อนเสมอ แล้ว assert ว่าไม่มี sample ไหนโดนตัด

2. prompt ตอนเทรนต้องตรงกับที่ Ollama render จริงแบบ token ต่อ token
   ollama_client ส่ง think=false ซึ่งทำให้ Ollama (0.34+) ทำสองอย่าง:
   เติม " /no_think" ท้ายข้อความ user และแทรก <think>\n\n</think>\n\n หลัง assistant
   → render_prompt() ทำแบบเดียวกัน (ตรวจแล้วว่า token ตรงกัน 5,667 = 5,667)
   ถ้าอัปเกรด Ollama หรือเปลี่ยนวิธีเรียก ต้องตรวจ parity ใหม่ทุกครั้ง

3. ref_model=None + peft_config ให้ PEFT ปิด adapter แล้วใช้ base เป็น reference
   ประหยัด VRAM ไปอีกก้อน (ไม่ต้องโหลดโมเดลที่สองมาเทียบ)

รันคำสั่ง:
    python train_dpo.py --dry-run     # วัด token อย่างเดียว ~1 นาที ไม่ใช้ GPU
    python train_dpo.py               # เทรนจริง
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "output" / "dpo"

THINK_BLOCK = "<think>\n\n</think>\n\n"
ASSISTANT_CUE = "<|im_start|>assistant\n"
# Ollama เติม suffix นี้ท้ายข้อความ user เองเมื่อเรียกด้วย think=false (ซ้อนกับ /no_think ที่
# build_prompt ใส่ไว้แล้ว) — ต้องเติมตามให้เหมือน ไม่งั้น prompt ตอนเทรนสั้นกว่าของจริง
OLLAMA_NO_THINK_SUFFIX = " /no_think"


def load_jsonl(path: Path) -> list[dict]:
    """อ่าน JSONL ทีละบรรทัด"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[int], pct: float) -> int:
    """คำนวณ percentile แบบง่าย (ไม่พึ่ง numpy)"""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct / 100))
    return ordered[idx]


def render_prompt(tok, system: str, user: str, strip_think: bool) -> str:
    """
    ประกอบ prompt ด้วย chat template จริงของโมเดล

    ไม่ปล่อยให้ TRL ทำเอง เพราะ TRL ไม่ส่ง enable_thinking ต่อให้ template
    และ Qwen3 ตั้ง default เป็น True ซึ่งจะได้ prefix คนละแบบกับตอนใช้งานจริง

    Args:
        tok: tokenizer ของโมเดล
        system: SYSTEM_PROMPT
        user: prompt ฝั่งผู้ใช้ (รวม /no_think ต่อท้ายแล้ว)
        strip_think: ตัด <think></think> ที่ template แทรกมาออกไหม (ปกติต้องเก็บไว้)

    Returns:
        ข้อความ prompt ที่ tokenize ได้เลย
    """
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user + OLLAMA_NO_THINK_SUFFIX},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if strip_think:
        text = text.replace(THINK_BLOCK, "")
    return text


def measure_lengths(rows: list[dict], tok, system: str, strip_think: bool) -> dict:
    """
    วัดความยาว token จริงของทั้งชุดข้อมูล ก่อนตั้งค่า max_length

    ต้องทำก่อนเทรนเสมอ — ถ้าตั้ง max_* ต่ำกว่าความจริง TRL จะตัด prompt ทิ้งแบบไม่แจ้งเตือน
    แล้วเราจะเทรนโมเดลให้ชอบ JSON ที่ขาดกลางคัน

    Returns:
        dict ของสถิติความยาว
    """
    prompt_len: list[int] = []
    completion_len: list[int] = []

    for row in rows:
        text = render_prompt(tok, system, row["user"], strip_think)
        prompt_len.append(len(tok(text, add_special_tokens=False)["input_ids"]))
        for key in ("chosen", "rejected"):
            # +1 เผื่อ EOS ที่ TRL เติมให้ท้ายคำตอบ
            completion_len.append(len(tok(row[key], add_special_tokens=False)["input_ids"]) + 1)

    return {
        "samples": len(rows),
        "prompt": {
            "p50": percentile(prompt_len, 50),
            "p95": percentile(prompt_len, 95),
            "max": max(prompt_len),
        },
        "completion": {
            "p50": percentile(completion_len, 50),
            "p95": percentile(completion_len, 95),
            "max": max(completion_len),
        },
    }


def resolve_limits(stats: dict) -> tuple[int, int, int]:
    """ตั้ง max_* จากความยาวจริง + เผื่อนิดหน่อย แล้วปัดขึ้นให้ลงตัว"""
    max_prompt = math.ceil(stats["prompt"]["max"] * 1.02 / 128) * 128
    max_completion = math.ceil(stats["completion"]["max"] * 1.05 / 64) * 64
    return max_prompt, max_completion, max_prompt + max_completion


def latest_checkpoint(out_dir: Path) -> Path | None:
    """
    หา checkpoint ล่าสุดที่เขียนเสร็จสมบูรณ์

    กรองด้วย trainer_state.json เพื่อข้าม checkpoint ที่เขียนค้างตอน session ถูกตัด
    """
    cks = [p for p in out_dir.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    if not cks:
        return None
    return max(cks, key=lambda p: int(p.name.split("-")[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tune SafeTrade บน H100")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True, help="ต้องเป็น storage ที่ไม่หายตอน session จบ")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dry-run", action="store_true", help="วัด token อย่างเดียว ไม่โหลดโมเดล ไม่เทรน")
    parser.add_argument(
        "--think-block",
        choices=["keep", "strip"],
        default="keep",
        help="keep = ตรงกับ Ollama ที่เรียกด้วย think=false (ค่าที่ถูกต้องตอนนี้)",
    )
    parser.add_argument("--attn", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--rpo-alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20250921)
    args = parser.parse_args()

    train_path = args.data_dir / "dpo_pairs_train.jsonl"
    eval_path = args.data_dir / "dpo_pairs_eval.jsonl"
    system_path = args.data_dir / "system_prompt.txt"
    manifest_path = args.data_dir / "dpo_manifest.json"

    for path in (train_path, system_path, manifest_path):
        if not path.exists():
            print(f"❌ ไม่พบไฟล์ {path} — รัน build_dpo_pairs.py ก่อน")
            return

    system = system_path.read_text(encoding="utf-8")
    system_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["system_sha256"] != system_sha:
        print("❌ system_prompt.txt ไม่ตรงกับ sha ใน manifest — ข้อมูลไม่ sync กัน")
        return

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path) if eval_path.exists() else []
    print(f"📂 train {len(train_rows)} pair | eval {len(eval_rows)} pair")

    # --- ตรวจเวอร์ชันไลบรารีก่อนแตะอะไรหนักๆ ---
    # debug บน LiCO แพง (ต้อง login เว็บใหม่ทุกรอบ) ให้พังตั้งแต่วินาทีแรกพร้อมบอกสาเหตุ
    import torch
    import transformers
    import trl
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print(f"ℹ️  torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__}")

    dpo_fields = {f.name for f in dataclasses.fields(DPOConfig)}
    required = {"beta", "max_length", "max_prompt_length"}
    missing = required - dpo_fields
    if missing:
        print(f"❌ DPOConfig ของ trl {trl.__version__} ไม่มี field {missing} — เวอร์ชันไม่ตรงกับที่สคริปต์รองรับ")
        return

    tok_kwarg = (
        "processing_class"
        if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters
        else "tokenizer"
    )
    dtype_kwarg = (
        "dtype"
        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        else "torch_dtype"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id == tok.eos_token_id:
        print("❌ pad_token เท่ากับ eos_token — TRL จะ mask EOS จริงทิ้ง โมเดลจะไม่เรียนรู้ที่จะหยุด")
        return

    strip_think = args.think_block == "strip"
    sample_prompt = render_prompt(tok, system, train_rows[0]["user"], strip_think)
    expected_tail = ASSISTANT_CUE if strip_think else ASSISTANT_CUE + THINK_BLOCK
    if not sample_prompt.endswith(expected_tail):
        print(
            f"❌ prompt ไม่ได้ลงท้ายด้วย {expected_tail!r} — template ของ tokenizer ไม่ตรงกับ Ollama\n"
            f"   ท้าย prompt: {sample_prompt[-120:]!r}"
        )
        return

    # --- วัดความยาวจริง ---
    print("📏 กำลังวัดความยาว token...")
    stats = measure_lengths(train_rows + eval_rows, tok, system, strip_think)
    max_prompt, max_completion, max_length = resolve_limits(stats)
    print(json.dumps(stats, indent=2))
    print(f"   → max_prompt_length={max_prompt} max_completion_length={max_completion} max_length={max_length}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "render_sample0.txt").write_text(sample_prompt, encoding="utf-8")

    if args.dry_run:
        print("✅ dry-run เสร็จ — ตรวจตัวเลขแล้วค่อยรันจริงโดยตัด --dry-run ออก")
        return

    from datasets import Dataset
    from peft import LoraConfig

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([
            {
                "prompt": render_prompt(tok, system, r["user"], strip_think),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ])

    train_ds = to_dataset(train_rows)
    eval_ds = to_dataset(eval_rows) if eval_rows else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **{dtype_kwarg: torch.bfloat16},
        attn_implementation=args.attn,
        device_map=None,  # ปล่อยให้ Trainer จัดการ — device_map="auto" จะตีกับ Trainer
    )
    model.config.use_cache = False  # ต้องปิดเมื่อใช้ gradient checkpointing
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # ไม่ใส่ modules_to_save (เช่น lm_head) — vocab 151,936 ทำให้ optimizer state บวมมาก
        # และทำให้เรื่อง reference model ไม่ชัดเจน
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    config_kwargs = dict(
        output_dir=str(args.output_dir),
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,  # NLL anchor — กัน logps/chosen ร่วงพร้อม rejected
        max_prompt_length=max_prompt,
        max_completion_length=max_completion,
        max_length=max_length,
        # TRL ตัด prompt จากด้านซ้ายเสมอ (ทิ้งหัว catalogue) ไม่ว่าตั้งค่านี้เป็นอะไร ตัวนี้มีผลแค่กับ
        # max_length — สิ่งที่กันจริงคือการตั้ง max_* จากความยาวที่วัดได้ + ตรวจ truncated ข้างล่าง
        truncation_mode="keep_start",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # DPO รวม chosen+rejected = 2 sequence ต่อ step อยู่แล้ว
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,  # default 1e-6 ของ DPOConfig ต่ำเกินไปสำหรับ LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",  # ไม่ยิงออกเน็ต — คลัสเตอร์มหาลัยมักบล็อกอยู่แล้ว
    )
    if eval_ds is not None:
        config_kwargs.update(eval_strategy="steps", eval_steps=args.save_steps)

    unknown = sorted(set(config_kwargs) - dpo_fields)
    if unknown:
        print(f"⚠️ trl {trl.__version__} ไม่รู้จัก config เหล่านี้ ข้ามให้: {unknown}")
    cfg = DPOConfig(**{k: v for k, v in config_kwargs.items() if k in dpo_fields})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # ใช้ base model (ปิด adapter) เป็น reference — ไม่ต้องโหลดโมเดลที่สอง
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        **{tok_kwarg: tok},
    )
    trainer.model.print_trainable_parameters()

    # ยืนยันว่าไม่มี sample ไหนโดนตัดจริงๆ หลัง tokenize แล้ว
    truncated = sum(
        1
        for row in trainer.train_dataset
        if len(row["prompt_input_ids"]) >= cfg.max_prompt_length
        or len(row["chosen_input_ids"]) >= cfg.max_completion_length
    )
    if truncated:
        print(f"❌ มี {truncated} sample ชนขอบ truncation — เพิ่ม max_* ก่อนเทรน ไม่งั้นข้อมูลขาด")
        return

    meta = {
        "model": args.model,
        "system_sha256": system_sha,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
        "token_stats": stats,
        "limits": {"max_prompt": max_prompt, "max_completion": max_completion, "max_length": max_length},
        "think_block": args.think_block,
        "pairs": {"train": len(train_rows), "eval": len(eval_rows)},
    }
    meta_path = args.output_dir / "run_meta.json"
    if meta_path.exists():
        # resume: ถ้า system prompt เปลี่ยนไประหว่างทาง adapter เดิมจะเข้ากันไม่ได้
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("system_sha256") != system_sha:
            print("❌ system_sha256 ไม่ตรงกับตอนเริ่มเทรนรอบก่อน — อย่า resume ทับ ให้เริ่ม output-dir ใหม่")
            return
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    ck = latest_checkpoint(args.output_dir)
    if ck:
        print(f"♻️ พบ checkpoint {ck.name} — เทรนต่อจากจุดนั้น")

    print("🚀 เริ่มเทรน")
    trainer.train(resume_from_checkpoint=str(ck) if ck else None)

    final = args.output_dir / "final_adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print(f"🎉 เทรนเสร็จ — adapter อยู่ที่ {final}")
    print("   ดาวน์โหลดโฟลเดอร์นี้กลับเครื่องทันที (ไฟล์เล็ก ~80MB และสร้างใหม่ไม่ได้)")
    print("ขั้นต่อไป: merge adapter → แปลง GGUF → ollama create")


if __name__ == "__main__":
    main()
