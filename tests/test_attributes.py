from aorus_assistant.rag.attributes import label_line, split_attributes


def test_split_keeps_parentheses():
    assert split_attributes("A (x, y), B, C (z)") == ["A (x, y)", "B", "C (z)"]


def test_display_line():
    facts = label_line("Display", "OLED WQXGA (2560×1600) 240Hz, 1ms, DCIP-3 100%, 500nits (peak), 1,000,000:1")
    assert "更新率 Refresh rate: 240Hz" in facts
    assert "解析度 Resolution: 2560×1600" in facts
    assert "面板 Panel: OLED WQXGA" in facts
    assert "亮度 Brightness: 500nits (peak)" in facts
    assert "對比度 Contrast ratio: 1,000,000:1" in facts
    assert label_line("Display", '16" 16:10') == ['螢幕尺寸 Screen size: 16"', "長寬比 Aspect ratio: 16:10"]


def test_cpu_line():
    facts = label_line("CPU", "Intel Core Ultra 9 Processor 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)")
    assert facts[0] == "型號 Model: Intel Core Ultra 9 Processor 275HX"
    assert "核心數 Cores: 24 cores" in facts and "最高時脈 Max clock: up to 5.4 GHz" in facts


def test_memory_weight_and_passthrough():
    assert label_line("System Memory", "Up to 64GB DDR5 5600MHz") == [
        "最大容量 Max capacity: Up to 64GB", "類型 Type: DDR5", "頻率 Speed: 5600MHz",
    ]
    assert label_line("Weight", "~2.5 kg") == ["重量 Weight: ~2.5 kg"]
    assert label_line("Color", "Dark Tide") == ["Dark Tide"]  # no rules -> verbatim
