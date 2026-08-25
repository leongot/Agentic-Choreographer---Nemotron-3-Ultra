"""
ingest_chroma.py — crea/rigenera il database vettoriale Chroma da aist_index.json.

Uso:
    python ingest_chroma.py

Opzionale:
    python ingest_chroma.py --reset

Gli embedding sono forzati su CPU per non occupare VRAM: il modello remoto OpenRouter non usa la GPU locale.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = Path("aist_index.json")
CHROMA_DIR = Path("./chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

GENRE_EXPANSION = {
    "gBR": "Breaking breakdance urban hip-hop acrobatica pavimento footwork freeze top rock power move potente energico underground strada dinamico esplosivo",
    "gPO": "Popping robotico scatti funk contrazione muscolare isolamento illusionismo preciso meccanico groove divertente",
    "gLO": "Locking funky espressivo stop ripartenze sorridente classico party allegro ritmato energico",
    "gMH": "Middle Hip-hop urban coreografico ritmo sincopato groovy vecchia scuola street potente controllato",
    "gLH": "LA style Hip-hop moderno commerciale fluido coreografico videoclip elegante pop morbido",
    "gHO": "House dance footwork gioco di gambe veloce fluido clubbing elettronico ipnotico groove leggero party",
    "gWA": "Waacking espressivo teatrale braccia veloci pose sfilata disco vogueing drammatico elegante emotivo",
    "gKR": "Krump aggressivo potente emotivo grezzo viscerale rilasci energetici intenso drammatico esplosivo",
    "gJB": "Ballet Jazz jazz ballet fusione classico moderno salti estensione fluidita lineare lirico elegante drammatico",
    "gJS": "Street Jazz commerciale moderno sensuale energico videoclip pop ritmato dinamico esuberante divertente teatrale",
}

DIFFICULTY_EXPANSION = {
    "facile": "basic semplice accessibile guidato base principiante",
    "difficile": "freestyle avanzato complesso intenso difficile virtuoso",
    "medio": "intermedio moderato bilanciato",
}

MOOD_BY_GENRE = {
    "gBR": "energico potente urbano atletico esplosivo tecnico dinamico",
    "gPO": "funky preciso robotico giocoso meccanico controllato",
    "gLO": "divertente funky allegro party sorridente ritmato",
    "gMH": "urban groove ritmato street controllato",
    "gLH": "fluido moderno commerciale elegante morbido",
    "gHO": "fluido veloce party groove leggero ipnotico",
    "gWA": "teatrale drammatico elegante emotivo espressivo",
    "gKR": "aggressivo potente drammatico emotivo intenso viscerale",
    "gJB": "lirico elegante drammatico fluido esteso",
    "gJS": "divertente sensuale pop energico moderno commerciale teatrale esuberante",
}


def load_dataset(index_path: Path = INDEX_PATH) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} non trovato. Genera prima aist_index.json.")
    with open(index_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, dict):
        raise ValueError("aist_index.json deve essere un dizionario {id: metadata}.")
    return dataset


def build_document(item: dict[str, Any]) -> Document:
    genre_code = item.get("genre_code", "")
    genre_name = item.get("genre_name", genre_code)
    bpm = int(item.get("bpm", 0) or 0)
    difficulty = item.get("difficulty", "medio")
    situation_code = item.get("situation_code", "")

    text_content = (
        f"ID: {item.get('id')}. "
        f"Stile: {genre_name} ({genre_code}). "
        f"Caratteristiche stile: {GENRE_EXPANSION.get(genre_code, '')}. "
        f"Mood compatibili: {MOOD_BY_GENRE.get(genre_code, '')}. "
        f"Ritmo: {bpm} BPM. "
        f"Difficoltà: {difficulty}. Situazione: {situation_code}. "
        f"Tipo movimento: {DIFFICULTY_EXPANSION.get(difficulty, '')}. "
        f"Musica: {item.get('music_code')}. Dancer: {item.get('dancer_id')}."
    )

    # Chroma accetta metadati scalari. Manteniamo tutti i campi utili a playlist.json.
    metadata = {
        "id": item.get("id"),
        "path": item.get("path"),
        "extension": item.get("extension"),
        "genre_code": genre_code,
        "genre_name": genre_name,
        "situation_code": situation_code,
        "music_code": item.get("music_code"),
        "bpm": bpm,
        "difficulty": difficulty,
        "dancer_id": item.get("dancer_id"),
    }

    # Rimuove solo chiavi con None, perché Chroma può rifiutare metadati None.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return Document(page_content=text_content, metadata=metadata)


def build_chroma_db(reset: bool = False) -> None:
    if reset and CHROMA_DIR.exists():
        print(f"Rimuovo database Chroma precedente: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    dataset = load_dataset(INDEX_PATH)
    docs = []

    print("Elaborazione dataset e generazione documenti Chroma...")
    for _, item in sorted(dataset.items()):
        if isinstance(item, dict) and item.get("id"):
            docs.append(build_document(item))

    if not docs:
        raise RuntimeError("Nessun documento valido trovato in aist_index.json.")

    print(f"Salvataggio nel database Chroma persistente: {CHROMA_DIR}")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Completato! Indicizzati {len(docs)} elementi su Chroma DB.")


if __name__ == "__main__":
    build_chroma_db(reset="--reset" in sys.argv)
