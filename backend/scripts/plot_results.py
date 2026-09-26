"""
SafeTrade — วาดกราฟผล Blind Test เป็นไฟล์ PNG

อ่าน output/grading_report.json แล้ววาดกราฟ 4 รูป สำหรับเอาไปใส่รายงาน/สไลด์

รันคำสั่ง:
    python plot_results.py
    python plot_results.py --dpi 300      # ความละเอียดสูงสำหรับงานพิมพ์

หมายเหตุเรื่องฟอนต์: matplotlib ไม่มีฟอนต์ไทยมาให้ ถ้าไม่ตั้งค่าข้อความไทยจะขึ้นเป็นสี่เหลี่ยม
สคริปต์นี้จึงไล่หาฟอนต์ไทยในเครื่อง (Tahoma / Leelawadee UI) มาใช้อัตโนมัติ

หลักการวาด (ตามแนวทาง data visualization):
- pattern id เป็นหมวดหมู่ที่ไม่มีลำดับ จึงใช้ "สีเดียวทุกแท่ง" ไม่ไล่เฉดตามค่า
  (การไล่เฉดจะเป็นการเข้ารหัสความยาวแท่งซ้ำซ้อน และกินช่องทางสีไปเปล่าๆ)
- ตัวอักษรใช้สีหมึกเสมอ ไม่ใช้สีของข้อมูล
- เส้นตารางบาง ทึบ ถอยไปเป็นพื้นหลัง ให้ข้อมูลเด่นที่สุด
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ไม่ต้องเปิดหน้าต่าง เขียนไฟล์อย่างเดียว

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import PATTERN_CATALOGUE  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REPORT = OUTPUT_DIR / "grading_report.json"
CHART_DIR = OUTPUT_DIR / "charts"

# พาเลตต์ — ผ่านการตรวจ colour-blind safety ทั้งโหมดสว่างและมืดแล้ว
SERIES_1 = "#2a78d6"  # น้ำเงิน — ซีรีส์หลัก
SERIES_2 = "#eb6834"  # ส้ม — ซีรีส์ที่สอง
INK = "#14171c"
INK_SOFT = "#5b6068"
GRID = "#e4e5e8"
SURFACE = "#ffffff"


def setup_thai_font() -> str:
    """
    หาฟอนต์ไทยในเครื่องแล้วตั้งเป็นค่าเริ่มต้นของ matplotlib

    ถ้าไม่เจอ ข้อความไทยจะแสดงเป็นสี่เหลี่ยม — จะเตือนให้รู้ตัว

    Returns:
        ชื่อ font family ที่ตั้งไว้ (หรือ "sans-serif" ถ้าไม่เจอฟอนต์ไทย)
    """
    candidates = [
        ("C:/Windows/Fonts/tahoma.ttf", "Tahoma"),
        ("C:/Windows/Fonts/leelawui.ttf", "Leelawadee UI"),
        ("C:/Windows/Fonts/LeelaUIb.ttf", "Leelawadee UI"),
    ]
    for path, name in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    print("[!] ไม่พบฟอนต์ไทยในเครื่อง — ข้อความไทยจะขึ้นเป็นสี่เหลี่ยม")
    return "sans-serif"


def style_axes(ax, xmax: float | None = None) -> None:
    """ตั้งหน้าตาแกนให้เรียบและถอยไปเป็นพื้นหลัง"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    if xmax is not None:
        ax.set_xlim(0, xmax)


def bar_labels(ax, bars, values, fmt: str = "{:.2f}", pad: float = 0.012) -> None:
    """ใส่ตัวเลขที่ปลายแท่ง (ตัวอักษรใช้สีหมึก ไม่ใช้สีข้อมูล)"""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )


def chart_f1(data: list[dict], dpi: int) -> Path:
    """กราฟหลัก — F1 รายตัว เรียงจากน้อยไปมาก ใช้สีเดียวทุกแท่ง"""
    rows = sorted(data, key=lambda p: p["f1"])
    fig, ax = plt.subplots(figsize=(9, 6.2))

    values = [p["f1"] for p in rows]
    bars = ax.barh([p["id"] for p in rows], values, color=SERIES_1, height=0.62)
    bar_labels(ax, bars, values)

    ax.axvline(0.5, color=INK_SOFT, linewidth=1, alpha=0.45)
    ax.text(0.508, len(rows) - 0.5, "0.50 = เริ่มใช้งานได้", fontsize=8, color=INK_SOFT)

    style_axes(ax, xmax=1.0)
    ax.set_xlabel("F1 score", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title(
        "F1 รายตัวของแต่ละ pattern (โมเดลล้วนๆ ก่อน fine-tune)",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )

    path = CHART_DIR / "1_f1_by_pattern.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_precision_recall(data: list[dict], dpi: int) -> Path:
    """สองซีรีส์ — ต้องมี legend เพราะมีมากกว่า 1 ซีรีส์"""
    rows = sorted(data, key=lambda p: p["r"])
    positions = range(len(rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9, 6.6))

    ax.barh(
        [i + height / 2 + 0.02 for i in positions],
        [p["p"] for p in rows],
        height=height,
        color=SERIES_1,
        label="Precision — ที่ตอบว่าเจอ ถูกกี่ส่วน",
    )
    ax.barh(
        [i - height / 2 - 0.02 for i in positions],
        [p["r"] for p in rows],
        height=height,
        color=SERIES_2,
        label="Recall — ที่มีจริง จับได้กี่ส่วน",
    )

    ax.set_yticks(list(positions))
    ax.set_yticklabels([p["id"] for p in rows])
    style_axes(ax, xmax=1.05)
    ax.set_xlabel("สัดส่วน", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title(
        "Precision เทียบ Recall — โมเดลตอบเกินหรือตอบขาด",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    path = CHART_DIR / "2_precision_recall.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_confusion(pairs: list[dict], dpi: int) -> Path:
    """คู่ที่โมเดลสับสนบ่อยที่สุด"""
    rows = sorted(pairs, key=lambda x: x["n"])
    labels = ["{} -> {}".format(x["gt"], x["pred"]) for x in rows]
    values = [x["n"] for x in rows]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels, values, color=SERIES_1, height=0.58)
    bar_labels(ax, bars, values, fmt="{:.0f}", pad=1.5)

    style_axes(ax, xmax=max(values) * 1.13)
    ax.set_xlabel("จำนวนครั้ง", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title(
        "เฉลยเป็นตัวซ้าย แต่โมเดลตอบตัวขวาแทน",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )

    path = CHART_DIR / "3_confusion_pairs.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_category(by_category: dict, dpi: int) -> Path:
    """คะแนนคลาดเฉลี่ยแยกตามหมวดของบทสนทนา"""
    names = {"fraud": "แชทโกง", "normal": "แชทปกติ", "suspicious": "น่าสงสัย"}
    rows = sorted(by_category.items(), key=lambda kv: kv[1]["final_score_mae"])
    values = [v["final_score_mae"] for _, v in rows]

    fig, ax = plt.subplots(figsize=(7.5, 3))
    bars = ax.barh([names[k] for k, _ in rows], values, color=SERIES_1, height=0.52)
    bar_labels(ax, bars, values, fmt="{:.1f}", pad=0.9)

    style_axes(ax, xmax=max(values) * 1.18)
    ax.set_xlabel("คะแนนคลาดเฉลี่ย (จุด จาก 100)", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title(
        "คะแนนสุดท้ายของแชทคลาดไปเท่าไหร่",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )

    path = CHART_DIR / "4_error_by_category.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="วาดกราฟผล blind test เป็นไฟล์ PNG")
    parser.add_argument("--input", type=Path, default=REPORT)
    parser.add_argument("--dpi", type=int, default=200, help="ความละเอียด (ใช้ 300 สำหรับงานพิมพ์)")
    args = parser.parse_args()

    if not args.input.exists():
        print("[X] ไม่พบไฟล์ {} — รัน grade_results.py ก่อน".format(args.input))
        return

    font = setup_thai_font()
    print("ฟอนต์ที่ใช้: {}".format(font))

    report = json.loads(args.input.read_text(encoding="utf-8"))
    per_model = report["model_only"]["patterns"]["per_pattern"]
    per_system = report["full_system"]["patterns"]["per_pattern"]

    data = [
        {
            "id": pid,
            "label": PATTERN_CATALOGUE[pid]["label"],
            "n": v["support"],
            "p": v["precision"],
            "r": v["recall"],
            "f1": v["f1"],
            "f1_sys": per_system[pid]["f1"],
        }
        for pid, v in per_model.items()
        if v["support"] > 0
    ]

    confusion = report["model_only"]["patterns"]["confusion"]
    pairs = sorted(
        [
            {"gt": gt, "pred": pred, "n": n}
            for gt, wrongs in confusion.items()
            for pred, n in wrongs.items()
        ],
        key=lambda x: -x["n"],
    )[:8]

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    made = [
        chart_f1(data, args.dpi),
        chart_precision_recall(data, args.dpi),
        chart_confusion(pairs, args.dpi),
        chart_category(report["by_category"], args.dpi),
    ]

    print("\nวาดเสร็จ {} รูป (dpi={})".format(len(made), args.dpi))
    for path in made:
        print("   {}  ({} KB)".format(path.name, path.stat().st_size // 1024))
    print("\nอยู่ในโฟลเดอร์: {}".format(CHART_DIR))


if __name__ == "__main__":
    main()
