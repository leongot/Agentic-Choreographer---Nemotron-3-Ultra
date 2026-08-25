"""
aist_index.py — costruisce/carica l'indice metadati AIST++.

Uso:
    python aist_index.py <cartella_dataset> [output.json]

Esempio:
    python aist_index.py . aist_index.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GENRE_MAP = {
    "gBR": {"name": "Breaking", "bpm_range": [80, 130]},
    "gPO": {"name": "Popping", "bpm_range": [80, 130]},
    "gLO": {"name": "Locking", "bpm_range": [80, 130]},
    "gMH": {"name": "Middle Hip-hop", "bpm_range": [80, 130]},
    "gLH": {"name": "LA style Hip-hop", "bpm_range": [80, 130]},
    "gHO": {"name": "House", "bpm_range": [110, 135]},
    "gWA": {"name": "Waacking", "bpm_range": [80, 130]},
    "gKR": {"name": "Krump", "bpm_range": [80, 130]},
    "gJB": {"name": "Ballet Jazz", "bpm_range": [80, 130]},
    "gJS": {"name": "Street Jazz", "bpm_range": [80, 130]},
}

KEYWORD_TO_GENRES = {
    "street jazz": ["gJS"],
    "jazz": ["gJS"],
    "breaking": ["gBR"],
    "break": ["gBR"],
    "breakdance": ["gBR"],
    "popping": ["gPO"],
    "pop": ["gPO"],
    "locking": ["gLO"],
    "lock": ["gLO"],
    "hip-hop": ["gMH", "gLH"],
    "hiphop": ["gMH", "gLH"],
    "hip hop": ["gMH", "gLH"],
    "house": ["gHO"],
    "waacking": ["gWA"],
    "waack": ["gWA"],
    "krump": ["gKR"],
    "krumping": ["gKR"],
    "jumpstyle": ["gJB"],
    "ballet jazz": ["gJB"],
    "energico": ["gBR", "gKR", "gJB"],
    "energica": ["gBR", "gKR", "gJB"],
    "aggressivo": ["gBR", "gKR"],
    "aggressiva": ["gBR", "gKR"],
    "potente": ["gBR", "gWA", "gKR"],
    "fluido": ["gHO", "gLH", "gMH"],
    "fluida": ["gHO", "gLH", "gMH"],
    "rilassato": ["gHO", "gLH"],
    "rilassata": ["gHO", "gLH"],
    "lento": ["gLH", "gJS"],
    "lenta": ["gLH", "gJS"],
    "veloce": ["gJB", "gBR", "gHO"],
    "drammatico": ["gWA", "gJS", "gKR"],
    "drammatica": ["gWA", "gJS", "gKR"],
    "elegante": ["gJS", "gWA", "gJB"],
    "funky": ["gLO", "gPO"],
    "urbano": ["gMH", "gBR"],
    "urbana": ["gMH", "gBR"],
    "festoso": ["gJB", "gLO"],
    "festosa": ["gJB", "gLO"],
    "meccanico": ["gPO"],
    "meccanica": ["gPO"],
    "preciso": ["gPO", "gLO"],
    "precisa": ["gPO", "gLO"],
    "teatrale": ["gWA", "gJS"],
}

COMPLEXITY_MAP = {
    "sBM": "facile",
    "sFM": "difficile",
}


def get_exact_bpm(genre_code: str, music_code: str) -> int:
    try:
        music_idx = int(re.search(r"\d+", music_code).group())
    except (AttributeError, ValueError):
        return 100

    if genre_code == "gHO":
        bpms = [110, 115, 120, 125, 130, 135]
    else:
        bpms = [80, 90, 100, 110, 120, 130]

    return bpms[music_idx] if music_idx < len(bpms) else bpms[-1]


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_index(dataset_root: str, output_path: str = "aist_index.json") -> dict:
    dataset_root_path = Path(dataset_root).resolve()
    index = {}

    pattern = re.compile(
        r"^(g[A-Z]{2})_(s[A-Z]{2})_c[A-Za-z]+_(d\d{2})_(m[A-Z0-9]+)_ch\d+\.(pkl|bvh|npy)$"
    )

    for fpath in sorted(dataset_root_path.rglob("*")):
        if fpath.suffix.lower() not in (".pkl", ".bvh", ".npy"):
            continue

        match = pattern.match(fpath.name)
        if not match:
            continue

        genre_code = match.group(1)
        situation_code = match.group(2)
        dancer_id = match.group(3)
        music_code = match.group(4)
        file_id = fpath.stem

        genre_info = GENRE_MAP.get(genre_code, {"name": genre_code})
        exact_bpm = get_exact_bpm(genre_code, music_code)

        index[file_id] = {
            "id": file_id,
            "path": _relative_or_absolute(fpath, dataset_root_path),
            "extension": fpath.suffix,
            "genre_code": genre_code,
            "genre_name": genre_info["name"],
            "situation_code": situation_code,
            "music_code": music_code,
            "bpm": exact_bpm,
            "difficulty": COMPLEXITY_MAP.get(situation_code, "medio"),
            "dancer_id": dancer_id,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[aist_index] Indicizzati {len(index)} file -> {output_path}")
    return index


def load_index(index_path: str = "aist_index.json") -> dict:
    path = Path(index_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python aist_index.py <cartella_dataset> [output.json]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else "aist_index.json"
    built = build_index(sys.argv[1], output)
    if built:
        print(json.dumps(next(iter(built.values())), indent=2, ensure_ascii=False))
