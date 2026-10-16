"""Attribute-level key-value extraction from spec value lines.

The spec page gives one value per field, often packing several attributes into a line:
    "OLED WQXGA (2560×1600) 240Hz, 1ms, DCIP-3 100%, 500nits (peak), 1,000,000:1"
A 0.8B model misreads these (answered "160Hz" / "1600 Hz" for the refresh rate), so each
line is split into attributes and recognised ones get explicit bilingual labels:
    更新率 Refresh rate: 240Hz
Rules are a maintainable table keyed by spec field; unmatched text is kept verbatim so no
information is lost.
"""

from __future__ import annotations

import re

# field_en -> [(pattern, label)]; patterns are tried in order on each attribute item.
RULES: dict[str, list[tuple[str, str]]] = {
    "Display": [
        (r'\d+(?:\.\d+)?"', "螢幕尺寸 Screen size"),
        (r"^[\d,]{5,}:1$", "對比度 Contrast ratio"),
        (r"(?<![\d,])\d{1,2}:\d{1,2}(?!\d)", "長寬比 Aspect ratio"),
        (r"\(?\d{3,4}\s*[×x]\s*\d{3,4}\)?", "解析度 Resolution"),
        (r"\d+\s*Hz", "更新率 Refresh rate"),
        (r"^\d+\s*ms$", "反應時間 Response time"),
        (r"DCI-?P-?3\s*\d+%", "色域 Color gamut"),
        (r"\d+\s*nits(?:\s*\(peak\))?", "亮度 Brightness"),
        (r"^(?:OLED|IPS|Mini ?LED)\b.*", "面板 Panel"),
    ],
    "System Memory": [
        (r"Up to \d+\s*GB", "最大容量 Max capacity"),
        (r"DDR\d", "類型 Type"),
        (r"\d+\s*MHz", "頻率 Speed"),
        (r"\d+x SO-DIMM sockets?", "插槽 Slots"),
    ],
    "CPU": [
        (r"\d+\s*cores", "核心數 Cores"),
        (r"\d+\s*threads", "執行緒 Threads"),
        (r"up to [\d.]+\s*GHz", "最高時脈 Max clock"),
        (r"\d+\s*MB cache", "快取 Cache"),
    ],
    "Video Graphics": [
        (r"^\d+\s*GB GDDR\d+$", "顯示記憶體 VRAM"),
        (r"^\d+\s*W Maximum Graphics Power.*", "最大顯卡功耗 TGP"),
        (r"^AI Boost.*", "加速時脈 Boost clock"),
        (r".*GeForce RTX.*", "顯示晶片 GPU"),
    ],
    "Storage": [
        (r"^Up to \d+\s*TB.*", "最大容量 Max capacity"),
        (r"^\d+x PCIe.*slots?$", "M.2 插槽 Slot"),
    ],
    "I/O Port": [
        (r"\d+ x MicroSD.*", "讀卡機 Card reader"),
        (r"\d+ x HDMI.*", "HDMI"),
        (r"\d+ x RJ-45.*", "有線網路孔 Ethernet port"),
        (r"\d+ x Audio Jack.*", "耳機麥克風孔 Audio jack"),
        (r"\d+ x DC in.*", "電源孔 DC-in"),
    ],
    "Audio": [(r"\d+x \d+W speakers", "喇叭 Speakers")],
    "Communications": [
        (r"WIFI \d.*", "無線網路 Wi-Fi"),
        (r"Bluetooth v?[\d.]+", "藍牙 Bluetooth"),
        (r"LAN: .*", "有線網路 Ethernet"),
    ],
    "Webcam": [(r".*Webcam$", "視訊鏡頭 Webcam")],
    "Battery": [(r"\d+\s*Wh", "電池容量 Capacity")],
    "Adapter": [(r"\d+\s*W", "變壓器功率 Power")],
    "Weight": [(r"~?[\d.]+\s*kg", "重量 Weight")],
    "Dimensions (W x D x H)": [(r".*mm$", "尺寸 寬x深x高 (W x D x H)")],
}


def split_attributes(line: str) -> list[str]:
    """Split on ', ' outside parentheses: '(36MB cache, up to 5.4 GHz)' stays intact."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(line):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and line.startswith(", ", i):
            parts.append(line[start:i])
            start = i + 2
    parts.append(line[start:])
    return [p.strip() for p in parts if p.strip()]


def _label_item(item: str, rules: list[tuple[str, str]]) -> list[str]:
    facts, rest = [], item
    for pattern, label in rules:
        m = re.search(pattern, rest, flags=re.IGNORECASE)
        if m:
            value = m.group(0).strip()
            if value.startswith("(") and value.endswith(")"):
                value = value[1:-1]
            facts.append(f"{label}: {value}")
            rest = (rest[: m.start()] + rest[m.end() :]).strip(" ,")
    rest = re.sub(r"\(\s*\)", "", rest).strip(" ,")
    return facts + ([rest] if rest else [])


def label_line(field_en: str, line: str) -> list[str]:
    rules = RULES.get(field_en, [])
    out: list[str] = []
    for item in split_attributes(line):
        if rules and "(" in item and field_en == "CPU":
            # "Intel Core Ultra 9 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)"
            name, _, inner = item.partition("(")
            out.append(f"型號 Model: {name.strip()}")
            for sub in split_attributes(inner.rstrip(")")):
                out.extend(_label_item(sub, rules))
        else:
            out.extend(_label_item(item, rules) if rules else [item])
    return out


def label_lines(field_en: str, lines: list[str]) -> list[str]:
    return [fact for line in lines for fact in label_line(field_en, line)]
