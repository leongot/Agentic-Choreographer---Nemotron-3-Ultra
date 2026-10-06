"""
tools.py — Pipeline coreografica AIST++
=======================================

Ruoli:
1. parse_dance_prompt   -> interpreta richiesta utente: genere, BPM, mood, difficoltà, numero mosse
2. search_aist_dataset  -> cerca SOLO nel database vettoriale Chroma e filtra sui metadati tecnici
3. build_sequence_plan  -> fa scegliere al modello OpenRouter la coreografia tra candidati reali e salva playlist.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from pathlib import Path
from typing import Any, Optional, Union

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aist_index import KEYWORD_TO_GENRES

logger = logging.getLogger(__name__)


# CONFIG
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHROMA_DIR = "./chroma_db"
PLAYLIST_FILENAME = "playlist.json"
MAX_CONTEXT_TOKENS = 4096          # limite logico usato per contenere i prompt
DEFAULT_NUM_SEQUENCES = 5
MAX_NUM_SEQUENCES = 20
SEARCH_K = 40                    # recupero ampio, poi filtro tecnico in Python
MAX_CANDIDATES_FOR_LLM = 10     # ridotto per evitare timeout sui modelli free OpenRouter

# Cache dell'ultima ricerca Chroma: serve se un caller passa candidati tagliati a build_sequence_plan.
LAST_CHROMA_RESULTS_LIST: list[dict[str, Any]] = []
LAST_CHROMA_RESULTS_BY_ID: dict[str, dict[str, Any]] = {}
LAST_PARSED_INTENT: dict[str, Any] = {}


# STATO WORKFLOW PER REACT SUPERVISIONATO
# Il ReAct agent può scegliere i tool, ma questo stato consente
# al controller di verificare se parse/search/build sono stati davvero eseguiti.
WORKFLOW_STATUS: dict[str, Any] = {
    "parsed": False,
    "searched": False,
    "built": False,
    # Prompt originale dell'utente. In modalità ReAct supervisionata resta bloccato
    # per evitare che il modello inventi una seconda richiesta e sovrascriva lo stato.
    "original_prompt": None,
    "original_n_sequenze": None,
    "last_prompt": None,
    "last_parse_json": None,
    "last_search_json": None,
    # Ultima ricerca Chroma realmente valida: il controller la preferisce sempre
    # rispetto a eventuali ricerche successive fallite/generate male da ReAct.
    "last_successful_search_json": None,
    "last_playlist_json": None,
    "ignored_react_prompts": [],
    "relaxed_filters_used": [],
    "react_error": None,
}


def reset_workflow_status(prompt: str = "") -> None:
    global WORKFLOW_STATUS
    expected_n = _bootstrap_extract_n_sequenze(prompt)
    WORKFLOW_STATUS = {
        "parsed": False,
        "searched": False,
        "built": False,
        "original_prompt": prompt,
        "original_n_sequenze": expected_n,
        "last_prompt": prompt,
        "last_parse_json": None,
        "last_search_json": None,
        "last_successful_search_json": None,
        "last_playlist_json": None,
        "ignored_react_prompts": [],
        "relaxed_filters_used": [],
        "react_error": None,
    }


def get_workflow_status() -> dict[str, Any]:
    return dict(WORKFLOW_STATUS)


def _set_workflow_status(**kwargs: Any) -> None:
    WORKFLOW_STATUS.update(kwargs)


def _bootstrap_extract_n_sequenze(prompt: str) -> Optional[int]:
    """Estrae subito il numero di sequenze dal prompt originale.

    Questa funzione sta prima di `estrai_n_sequenze_da_testo` solo perché serve già
    nel reset dello stato. Usa pattern semplici e difensivi.
    """
    testo = (prompt or "").lower()
    testo = unicodedata.normalize("NFD", testo)
    testo = "".join(c for c in testo if unicodedata.category(c) != "Mn")
    patterns = [
        r"\b(\d+)\s*(mosse|sequenze|passi|movimenti|clip)\b",
        r"\bda\s*(\d+)\s*(mosse|sequenze|passi|movimenti|clip)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, testo)
        if match:
            try:
                return max(1, min(int(match.group(1)), MAX_NUM_SEQUENCES))
            except (TypeError, ValueError):
                return None
    return None


def _get_locked_prompt(candidate_prompt: str = "") -> str:
    """Restituisce il prompt originale della sessione se esiste.

    Serve a impedire che ReAct, dopo una prima ricerca corretta, inventi una nuova
    richiesta e sovrascriva parse/search con risultati incoerenti.
    Fuori dalla modalità supervisionata, original_prompt è None e quindi viene usato
    normalmente il prompt passato al tool.
    """
    original_prompt = (WORKFLOW_STATUS.get("original_prompt") or "").strip()
    candidate_prompt = (candidate_prompt or "").strip()
    if original_prompt:
        if candidate_prompt and candidate_prompt != original_prompt:
            ignored = list(WORKFLOW_STATUS.get("ignored_react_prompts") or [])
            ignored.append(candidate_prompt)
            WORKFLOW_STATUS["ignored_react_prompts"] = ignored
            logger.warning(
                "Prompt ReAct ignorato perché diverso dal prompt originale. originale=%r, ricevuto=%r",
                original_prompt,
                candidate_prompt,
            )
        return original_prompt
    return candidate_prompt


def _get_forced_n_sequenze(candidate_n: Any = None) -> int:
    """Restituisce il numero di sequenze bloccato dal prompt originale.

    Priorità:
    1. numero scritto nel prompt originale dell'utente;
    2. numero già salvato nello stato workflow;
    3. numero passato dal tool;
    4. default.

    Questo impedisce a ReAct di chiamare build_sequence_plan con 5 quando l'utente
    aveva scritto 7 mosse.
    """
    original_prompt = WORKFLOW_STATUS.get("original_prompt") or ""
    explicit_from_prompt = _bootstrap_extract_n_sequenze(original_prompt)
    if explicit_from_prompt is not None:
        WORKFLOW_STATUS["original_n_sequenze"] = explicit_from_prompt
        return explicit_from_prompt

    forced = WORKFLOW_STATUS.get("original_n_sequenze")
    if forced is not None:
        try:
            return max(1, min(int(forced), MAX_NUM_SEQUENCES))
        except (TypeError, ValueError):
            pass

    try:
        return max(1, min(int(candidate_n or DEFAULT_NUM_SEQUENCES), MAX_NUM_SEQUENCES))
    except (TypeError, ValueError):
        return DEFAULT_NUM_SEQUENCES


def _store_expected_n_sequenze(n_sequenze: Any) -> None:
    """Salva il numero sequenze atteso, senza sovrascriverlo con default ReAct errati."""
    original_prompt = WORKFLOW_STATUS.get("original_prompt") or ""
    explicit_from_prompt = _bootstrap_extract_n_sequenze(original_prompt)
    if explicit_from_prompt is not None:
        WORKFLOW_STATUS["original_n_sequenze"] = explicit_from_prompt
        return

    try:
        n = max(1, min(int(n_sequenze or DEFAULT_NUM_SEQUENCES), MAX_NUM_SEQUENCES))
    except (TypeError, ValueError):
        n = DEFAULT_NUM_SEQUENCES

    if WORKFLOW_STATUS.get("original_prompt") and WORKFLOW_STATUS.get("original_n_sequenze") is None:
        WORKFLOW_STATUS["original_n_sequenze"] = n
    elif not WORKFLOW_STATUS.get("original_prompt"):
        WORKFLOW_STATUS["original_n_sequenze"] = n


def _store_search_status(payload: dict[str, Any]) -> str:
    """Salva lo stato della ricerca e preserva l'ultima ricerca valida."""
    search_payload_json = json.dumps(payload, ensure_ascii=False)
    update = {
        "searched": True,
        "last_search_json": search_payload_json,
    }
    if payload.get("success") and int(payload.get("totale_trovati") or 0) > 0:
        update["last_successful_search_json"] = search_payload_json
    if payload.get("filtri_rilassati"):
        update["relaxed_filters_used"] = payload.get("filtri_rilassati")
    _set_workflow_status(**update)
    return search_payload_json

# MAPPE AIST++
GENRE_CODE_TO_NAME = {
    "gBR": "Breaking",
    "gPO": "Popping",
    "gLO": "Locking",
    "gMH": "Middle Hip-hop",
    "gLH": "LA Hip-hop",
    "gHO": "House",
    "gWA": "Waacking",
    "gKR": "Krump",
    "gJB": "Ballet Jazz",
    "gJS": "Street Jazz",
}

SITUATION_TO_DIFFICULTY = {
    "sBM": "facile",
    "sFM": "difficile",
}

# Keyword esplicite di stile. Se l'utente specifica uno stile, NON si aggiungono generi da mood.
EXPLICIT_GENRE_KEYWORDS = [
    ("street jazz", ["gJS"]),
    ("jazz street", ["gJS"]),
    ("ballet jazz", ["gJB"]),
    ("breaking", ["gBR"]),
    ("breakdance", ["gBR"]),
    ("break dance", ["gBR"]),
    ("popping", ["gPO"]),
    ("locking", ["gLO"]),
    ("middle hip hop", ["gMH"]),
    ("middle hip-hop", ["gMH"]),
    ("la hip hop", ["gLH"]),
    ("la hip-hop", ["gLH"]),
    ("hip hop", ["gMH", "gLH"]),
    ("hip-hop", ["gMH", "gLH"]),
    ("house", ["gHO"]),
    ("waacking", ["gWA"]),
    ("waack", ["gWA"]),
    ("krump", ["gKR"]),
    ("krumping", ["gKR"]),
    ("jumpstyle", ["gJB"]),
    ("jazz", ["gJS"]),
]

MOOD_SYNONYMS = {
    "divertente": ["divertente", "giocoso", "giocosa", "allegro", "allegra", "festoso", "festosa", "leggero", "party", "playful"],
    "giocoso": ["giocoso", "giocosa", "divertente", "allegro", "allegra", "festoso", "party", "playful"],
    "giocosa": ["giocosa", "giocoso", "divertente", "allegro", "allegra", "festoso", "party", "playful"],
    "allegro": ["allegro", "allegra", "gioioso", "gioiosa", "divertente", "solare", "positivo"],
    "allegra": ["allegra", "allegro", "gioiosa", "gioioso", "divertente", "solare", "positivo"],
    "felice": ["felice", "allegro", "positivo", "solare", "gioioso"],
    "energica": ["energica", "energico", "energia", "dinamico", "potente", "esplosivo", "ritmato"],
    "energico": ["energica", "energico", "energia", "dinamico", "potente", "esplosivo", "ritmato"],
    "potente": ["potente", "forte", "fisico", "intenso", "esplosivo", "impatto"],
    "drammatica": ["drammatica", "drammatico", "teatrale", "intenso", "emotivo", "espressivo"],
    "drammatico": ["drammatica", "drammatico", "teatrale", "intenso", "emotivo", "espressivo"],
    "emotiva": ["emotiva", "emotivo", "sentito", "espressivo", "intenso", "lirico"],
    "emotivo": ["emotiva", "emotivo", "sentito", "espressivo", "intenso", "lirico"],
    "fluida": ["fluida", "fluido", "morbido", "continuo", "smooth", "elegante"],
    "fluido": ["fluida", "fluido", "morbido", "continuo", "smooth", "elegante"],
    "aggressiva": ["aggressiva", "aggressivo", "duro", "potente", "raw", "viscerale"],
    "aggressivo": ["aggressiva", "aggressivo", "duro", "potente", "raw", "viscerale"],
    "sensuale": ["sensuale", "commerciale", "pop", "videoclip", "espressivo"],
    "elegante": ["elegante", "raffinato", "lineare", "teatrale", "espressivo"],
    "funky": ["funky", "groove", "ritmato", "sincopato", "divertente"],
    "urbana": ["urbana", "urbano", "street", "underground", "hip-hop"],
    "urbano": ["urbana", "urbano", "street", "underground", "hip-hop"],
}

# Correzioni di refusi frequenti prima di interpretare mood/stili.
TYPO_FIXES = {
    "energitca": "energica",
    "energitcha": "energica",
    "divertnete": "divertente",
    "drammtica": "drammatica",
    "dramatico": "drammatico",
    "strett jazz": "street jazz",
    "streetjazz": "street jazz",
    "braking": "breaking",
}

# CHROMA
embeddings_chroma = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vector_db = (
    Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings_chroma)
    if Path(CHROMA_DIR).exists()
    else None
)



# MODELLI LLM
def crea_llm_tool(
    temperature: float,
    reasoning: bool = False,
    num_predict: int = 1024,
) -> ChatOpenAI:
    """Crea un LLM remoto via OpenRouter usando Nemotron free.

    La variabile reasoning resta nel parametro per compatibilità con il resto del codice,
    ma non viene inviata: i modelli OpenRouter non accettano tutti le stesse opzioni extra.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY non trovata. Impostala con: "
            'setx OPENROUTER_API_KEY "la_tua_api_key"'
        )

    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=num_predict,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        timeout=120,
        default_headers={
            "HTTP-Referer": "http://127.0.0.1:5000",
            "X-Title": "Agentic Choreography Thesis Project",
        },
    )



# HELPER NORMALIZZAZIONE TESTO
def normalizza_testo(testo: str) -> str:
    testo = testo or ""
    testo = testo.lower().strip()
    testo = "".join(
        c for c in unicodedata.normalize("NFD", testo)
        if unicodedata.category(c) != "Mn"
    )
    for sbagliato, corretto in TYPO_FIXES.items():
        testo = testo.replace(sbagliato, corretto)
    testo = re.sub(r"\s+", " ", testo)
    return testo


def estrai_n_sequenze_da_testo(prompt: str, default: int = DEFAULT_NUM_SEQUENCES) -> int:
    testo = normalizza_testo(prompt)
    patterns = [
        r"\b(\d+)\s*(mosse|sequenze|passi|movimenti|clip)\b",
        r"\bda\s*(\d+)\s*(mosse|sequenze|passi|movimenti|clip)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, testo)
        if match:
            return max(1, min(int(match.group(1)), MAX_NUM_SEQUENCES))
    return default


def normalizza_generi(prompt: str, genre_codes: Optional[list[str]] = None) -> list[str]:
    testo = normalizza_testo(prompt)
    espliciti: set[str] = set()

    for keyword, codes in EXPLICIT_GENRE_KEYWORDS:
        if keyword in testo:
            # Le keyword sono ordinate dalla più specifica alla più generale.
            # Appena troviamo uno stile esplicito, lo facciamo vincere sui mood
            # e impediamo che "ballet jazz" venga allargato anche a "jazz".
            return sorted(set(codes))

    found: set[str] = set(genre_codes or [])

    # Fallback: se non c'è uno stile esplicito, i mood possono suggerire generi compatibili.
    for keyword, codes in KEYWORD_TO_GENRES.items():
        if keyword in testo:
            found.update(codes)

    return sorted(found)


def estrai_bpm_da_testo(
    prompt: str,
    min_bpm: Optional[int] = None,
    max_bpm: Optional[int] = None,
    tempo_description: Optional[str] = None,
) -> tuple[Optional[int], Optional[int], Optional[str]]:
    testo = normalizza_testo(prompt)

    # Range: "tra 90 e 110 bpm", "da 90 a 110 bpm", "90-110 bpm", "90/110 bpm"
    range_patterns = [
        r"(?:tra|fra|da|dai|dalle)\s*(\d{2,3})\s*(?:e|a|ai|alle|-)\s*(\d{2,3})\s*bpm",
        r"\b(\d{2,3})\s*[-/]\s*(\d{2,3})\s*bpm\b",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, testo)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            return min(a, b), max(a, b), tempo_description

    # Minimo: "sopra 120 bpm", "oltre 120 bpm", "almeno 120 bpm"
    match = re.search(r"(?:sopra|oltre|almeno|minimo|min\.?|piu di|piu\s+di)\s*(\d{2,3})\s*bpm", testo)
    if match:
        min_bpm = int(match.group(1))

    # Massimo: "sotto 100 bpm", "meno di 100 bpm", "massimo 100 bpm"
    match = re.search(r"(?:sotto|meno di|meno\s+di|massimo|max\.?)\s*(\d{2,3})\s*bpm", testo)
    if match:
        max_bpm = int(match.group(1))

    # BPM singolo: "100 bpm" -> valore esatto, solo se non già coperto da sopra/sotto/range.
    if min_bpm is None and max_bpm is None:
        match = re.search(r"\b(\d{2,3})\s*bpm\b", testo)
        if match:
            value = int(match.group(1))
            return value, value, tempo_description

    # Descrizioni tempo, usate solo se non ci sono numeri BPM espliciti.
    if min_bpm is None and max_bpm is None:
        if "medio veloce" in testo or "medio-veloce" in testo:
            return 110, 135, "medio-veloce"
        if "molto veloce" in testo or "velocissima" in testo or "velocissimo" in testo:
            return 125, None, "veloce"
        if "veloce" in testo or "alta energia" in testo or "ritmo alto" in testo:
            return 120, None, "veloce"
        if "moderato" in testo or "ritmo medio" in testo:
            return 90, 120, "moderato"
        if "lento" in testo or "lenta" in testo:
            return None, 105, "lento"

    return min_bpm, max_bpm, tempo_description


def utente_ha_chiesto_bpm_o_tempo(prompt: str) -> bool:
    """
    True solo se l'utente ha chiesto esplicitamente un vincolo BPM/tempo.

    Regola importante per il progetto:
    - "energica", "energico", "potente", "divertente" sono mood/qualità espressive;
    - non devono diventare automaticamente min_bpm/max_bpm;
    - il BPM è un filtro tecnico solo quando l'utente scrive BPM o parole di tempo esplicite.
    """
    testo = normalizza_testo(prompt)

    # Vincoli numerici espliciti: "120 bpm", "90-110 bpm", "tra 90 e 110 bpm".
    if "bpm" in testo:
        return True

    # Parole che esprimono davvero tempo/velocità, non semplice energia/mood.
    parole_tempo = [
        "veloce",
        "velocissima",
        "velocissimo",
        "lento",
        "lenta",
        "lentissima",
        "lentissimo",
        "moderato",
        "moderata",
        "ritmo medio",
        "ritmo alto",
        "ritmo basso",
        "medio veloce",
        "medio-veloce",
        "molto veloce",
    ]
    return any(parola in testo for parola in parole_tempo)


def normalizza_difficulty(prompt: str, difficulty: Optional[str] = None) -> Optional[str]:
    testo = normalizza_testo(prompt)
    if any(x in testo for x in ["facile", "semplice", "base", "basic", "principiante"]):
        return "facile"
    if any(x in testo for x in ["difficile", "avanzata", "avanzato", "freestyle", "complessa", "dura"]):
        return "difficile"
    return difficulty


def normalizza_mood(prompt: str, mood_keywords: Optional[list[str]] = None) -> list[str]:
    testo = normalizza_testo(prompt)
    mood_finali: set[str] = set()

    for mood in mood_keywords or []:
        if mood:
            mood_norm = normalizza_testo(mood)
            if mood_norm:
                mood_finali.add(mood_norm)

    for mood, sinonimi in MOOD_SYNONYMS.items():
        if mood in testo:
            mood_finali.update(sinonimi)

    # Piccolo fallback: conserva aggettivi frequenti anche se non erano in tabella.
    for token in re.findall(r"\b[a-z]{5,}\b", testo):
        if token in MOOD_SYNONYMS:
            mood_finali.update(MOOD_SYNONYMS[token])

    return sorted(mood_finali)



# HELPER METADATI AIST
def estrai_metadati_da_id(seq_id: str) -> dict[str, Any]:
    metadati: dict[str, Any] = {}
    if not seq_id or not isinstance(seq_id, str):
        return metadati

    parti = seq_id.split("_")
    if len(parti) >= 1:
        metadati["genre_code"] = parti[0]
        metadati["genre_name"] = GENRE_CODE_TO_NAME.get(parti[0], parti[0])
    if len(parti) >= 2:
        metadati["situation_code"] = parti[1]
        metadati["difficulty"] = SITUATION_TO_DIFFICULTY.get(parti[1], "medio")
    if len(parti) >= 3:
        metadati["camera_code"] = parti[2]

    for parte in parti:
        if re.fullmatch(r"d\d{2}", parte):
            metadati["dancer_id"] = parte
        elif re.fullmatch(r"m[A-Z]+\d+", parte):
            metadati["music_code"] = parte
        elif re.fullmatch(r"ch\d+", parte):
            metadati["sequence_code"] = parte

    metadati["path"] = f"motions\\{seq_id}.pkl"
    return metadati


def normalizza_metadati_candidato(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}

    candidato = dict(item)
    seq_id = candidato.get("id")
    if seq_id:
        for chiave, valore in estrai_metadati_da_id(seq_id).items():
            if candidato.get(chiave) is None:
                candidato[chiave] = valore

    if candidato.get("genre_name") is None and candidato.get("genre_code"):
        candidato["genre_name"] = GENRE_CODE_TO_NAME.get(candidato.get("genre_code"), candidato.get("genre_code"))
    if candidato.get("genere") is None:
        candidato["genere"] = candidato.get("genre_name")
    if candidato.get("difficulty") is None and candidato.get("situation_code"):
        candidato["difficulty"] = SITUATION_TO_DIFFICULTY.get(candidato.get("situation_code"), "medio")

    if candidato.get("bpm") is not None:
        try:
            candidato["bpm"] = int(candidato["bpm"])
        except (TypeError, ValueError):
            pass

    return candidato


def salva_cache_chroma(risultati: list[dict[str, Any]]) -> None:
    global LAST_CHROMA_RESULTS_LIST, LAST_CHROMA_RESULTS_BY_ID
    LAST_CHROMA_RESULTS_LIST = [normalizza_metadati_candidato(x) for x in risultati if isinstance(x, dict)]
    LAST_CHROMA_RESULTS_BY_ID = {
        item["id"]: item
        for item in LAST_CHROMA_RESULTS_LIST
        if item.get("id")
    }


def completa_candidato_da_chroma(candidato: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(candidato, dict):
        return {}
    seq_id = candidato.get("id")
    if seq_id and seq_id in LAST_CHROMA_RESULTS_BY_ID:
        completo = dict(LAST_CHROMA_RESULTS_BY_ID[seq_id])
        for chiave, valore in candidato.items():
            if valore is not None:
                completo[chiave] = valore
        return normalizza_metadati_candidato(completo)
    return normalizza_metadati_candidato(candidato)


def filtra_candidati_tecnici(
    candidati: list[dict[str, Any]],
    genre_codes: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    min_bpm: Optional[int] = None,
    max_bpm: Optional[int] = None,
) -> list[dict[str, Any]]:
    genre_set = set(genre_codes or [])
    filtrati: list[dict[str, Any]] = []

    for item in candidati:
        item = normalizza_metadati_candidato(item)

        if genre_set and item.get("genre_code") not in genre_set:
            continue

        if difficulty == "facile" and item.get("situation_code") != "sBM":
            continue
        if difficulty == "difficile" and item.get("situation_code") != "sFM":
            continue

        bpm = item.get("bpm")
        try:
            bpm_int = int(bpm) if bpm is not None else None
        except (TypeError, ValueError):
            bpm_int = None

        if bpm_int is not None:
            if min_bpm is not None and bpm_int < min_bpm:
                continue
            if max_bpm is not None and bpm_int > max_bpm:
                continue

        filtrati.append(item)

    return filtrati


def deduplica_per_id(candidati: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in candidati:
        seq_id = item.get("id")
        if not seq_id or seq_id in seen:
            continue
        seen.add(seq_id)
        out.append(item)
    return out


def diversifica_candidati(candidati: list[dict[str, Any]], limite: int) -> list[dict[str, Any]]:
    """Mantiene ranking Chroma ma evita, se possibile, troppi duplicati dello stesso dancer/music."""
    candidati = deduplica_per_id([normalizza_metadati_candidato(x) for x in candidati])
    scelti: list[dict[str, Any]] = []
    usati_dancer_music: set[tuple[Any, Any]] = set()

    for item in candidati:
        key = (item.get("dancer_id"), item.get("music_code"))
        if key not in usati_dancer_music:
            scelti.append(item)
            usati_dancer_music.add(key)
        if len(scelti) >= limite:
            return scelti

    for item in candidati:
        if item.get("id") not in {x.get("id") for x in scelti}:
            scelti.append(item)
        if len(scelti) >= limite:
            break

    return scelti


# SCHEMI Pydantic
class DanceIntent(BaseModel):
    genre_codes: list[str] = Field(default=[], description="Codici genere AIST++, es. gBR, gHO, gJS")
    difficulty: Optional[str] = Field(default=None, description="facile, medio o difficile")
    motion_type: Optional[str] = Field(default=None, description="basic o freestyle")
    min_bpm: Optional[int] = Field(default=None, description="BPM minimo")
    max_bpm: Optional[int] = Field(default=None, description="BPM massimo")
    tempo_description: Optional[str] = Field(default=None, description="lento, moderato, medio-veloce, veloce")
    mood_keywords: list[str] = Field(default=[], description="Mood/aggettivi emotivi")
    n_sequenze: Optional[int] = Field(default=None, description="Numero di mosse/sequenze richieste")


class SequenzaScelta(BaseModel):
    posizione: int = Field(description="Posizione nella coreografia finale, partendo da 1")
    indice_candidato: int = Field(description="Indice del candidato scelto dalla lista fornita")
    motivo: str = Field(description="Motivo coreografico breve")


class SequencePlan(BaseModel):
    sequenze_ordinate: list[SequenzaScelta] = Field(description="Sequenze scelte dal modello tramite indici")
    descrizione_coreografia: str = Field(description="Descrizione narrativa della coreografia")
    durata_stimata_sec: int = Field(description="Durata stimata totale in secondi")



# TOOL 1 — PARSE
@tool
def parse_dance_prompt(prompt: str) -> str:
    """
    Interpreta la richiesta utente e restituisce un JSON con:
    genre_codes, mood_keywords, min_bpm, max_bpm, difficulty, tempo_description, n_sequenze.
    Usa il modello OpenRouter per comprensione linguistica e poi normalizzazione deterministica Python.
    """
    global LAST_PARSED_INTENT

    prompt = _get_locked_prompt(prompt)
    base = DanceIntent()

    try:
        llm = crea_llm_tool(temperature=0, reasoning=False, num_predict=300)

        prompt_interno = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Sei un parser JSON per richieste di danza AIST++.\n"
                        "Rispondi SOLO con un oggetto JSON, senza markdown.\n"
                        "Campi: genre_codes, difficulty, motion_type, min_bpm, max_bpm, "
                        "tempo_description, mood_keywords, n_sequenze.\n"
                        "Codici: gBR=Breaking, gPO=Popping, gLO=Locking, gMH=Middle Hip-hop, "
                        "gLH=LA Hip-hop, gHO=House, gWA=Waacking, gKR=Krump, "
                        "gJB=Ballet Jazz/Jumpstyle, gJS=Street Jazz.\n"
                        "street jazz -> gJS; breaking/breakdance -> gBR; se trovi '7 mosse', n_sequenze=7.\n"
                        "Non inventare BPM da mood.\n"
                        "min_bpm/max_bpm devono essere valorizzati solo se l'utente scrive BPM, veloce, lento, moderato o ritmo alto/medio/basso.\n"
                        "Formato esempio: {{\"genre_codes\":[\"gJS\"],\"difficulty\":null,\"motion_type\":null,\"min_bpm\":null,\"max_bpm\":null,\"tempo_description\":null,\"mood_keywords\":[\"energica\"],\"n_sequenze\":7}}.\n"
                        "Rispondi sempre e solo con JSON valido."
                    ),
                ),
                ("user", "Richiesta: {prompt}"),
            ]
        )

        logger.info("DEBUG [parse_dance_prompt]: invio la richiesta a OpenRouter......")
        risposta = llm.invoke(prompt_interno.format_messages(prompt=prompt))
        contenuto = getattr(risposta, "content", risposta)
        raw = _estrai_json_da_testo(contenuto)
        if isinstance(raw, dict):
            base = DanceIntent(**raw)
        logger.info("DEBUG [parse_dance_prompt]: risposta ricevuta da OpenRouter..")

    except Exception as e:
        logger.warning(f"[parse_dance_prompt] OpenRouter/parser non disponibile o output non valido. Uso fallback Python. Dettaglio: {e}")

    genre_codes = normalizza_generi(prompt, base.genre_codes)
    if utente_ha_chiesto_bpm_o_tempo(prompt):
        min_bpm, max_bpm, tempo_description = estrai_bpm_da_testo(
            prompt,
            base.min_bpm,
            base.max_bpm,
            base.tempo_description,
        )
    else:
        # Protezione: il modello può inferire "energica -> 120-140 BPM",
        # ma se l'utente non ha chiesto un vincolo di tempo/BPM lo azzeriamo.
        if base.min_bpm is not None or base.max_bpm is not None or base.tempo_description:
            logger.info(
                "DEBUG [parse_dance_prompt]: BPM/tempo suggeriti dal modello ignorati perché non espliciti nel prompt. "
                "min_bpm=%r max_bpm=%r tempo=%r",
                base.min_bpm,
                base.max_bpm,
                base.tempo_description,
            )
        min_bpm, max_bpm, tempo_description = None, None, None
    difficulty = normalizza_difficulty(prompt, base.difficulty)
    mood_keywords = normalizza_mood(prompt, base.mood_keywords)
    n_sequenze = estrai_n_sequenze_da_testo(prompt, base.n_sequenze or DEFAULT_NUM_SEQUENCES)

    payload = {
        "genre_codes": genre_codes,
        "difficulty": difficulty,
        "motion_type": base.motion_type,
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "tempo_description": tempo_description,
        "mood_keywords": mood_keywords,
        "n_sequenze": n_sequenze,
        "prompt_originale": prompt,
    }

    LAST_PARSED_INTENT = payload
    parsed_payload_json = json.dumps(payload, ensure_ascii=False)
    _store_expected_n_sequenze(n_sequenze)
    _set_workflow_status(
        parsed=True,
        last_prompt=prompt,
        last_parse_json=parsed_payload_json,
    )
    logger.info(f"DEBUG [parse_dance_prompt]: intent finale={payload}")
    return parsed_payload_json



# TOOL 2 — SEARCH CHROMA
@tool
def search_aist_dataset(
    genre_codes: list[str],
    mood_keywords: list[str] = [],
    difficulty: Optional[str] = None,
    min_bpm: Optional[int] = None,
    max_bpm: Optional[int] = None,
    tempo_description: Optional[str] = None,
    prompt_originale: str = "",
    n_sequenze: int = DEFAULT_NUM_SEQUENCES,
) -> str:
    """
    Cerca nel dataset vettorializzato Chroma in base alla richiesta.
    - Mood e prompt entrano nella query semantica.
    - Genere, BPM e difficoltà sono filtri tecnici sui metadati Chroma.
    """
    prompt_originale = _get_locked_prompt(prompt_originale or LAST_PARSED_INTENT.get("prompt_originale", ""))

    if vector_db is None:
        payload = {
            "success": False,
            "error": "Database Chroma non inizializzato. Esegui prima: python ingest_chroma.py",
            "risultati": [],
            "totale_trovati": 0,
        }
        return _store_search_status(payload)

    # Normalizzazione difensiva nel caso il tool sia chiamato direttamente.
    genre_codes = normalizza_generi(prompt_originale, genre_codes)
    bpm_richiesto_esplicitamente = utente_ha_chiesto_bpm_o_tempo(prompt_originale)
    if bpm_richiesto_esplicitamente:
        min_bpm, max_bpm, tempo_description = estrai_bpm_da_testo(
            prompt_originale,
            min_bpm,
            max_bpm,
            tempo_description,
        )
    else:
        # Protezione ulteriore: ReAct/LLM può passare min_bpm/max_bpm inventati.
        # Se l'utente non ha chiesto BPM/tempo, questi filtri tecnici vengono ignorati.
        if min_bpm is not None or max_bpm is not None or tempo_description:
            logger.info(
                "DEBUG [search_aist_dataset]: BPM/tempo ricevuti ma ignorati perché non espliciti nel prompt. "
                "min_bpm=%r max_bpm=%r tempo=%r",
                min_bpm,
                max_bpm,
                tempo_description,
            )
        min_bpm, max_bpm, tempo_description = None, None, None
    difficulty = normalizza_difficulty(prompt_originale, difficulty)
    mood_keywords = normalizza_mood(prompt_originale, mood_keywords)
    filtri_rilassati: list[str] = []

    genre_names = [GENRE_CODE_TO_NAME.get(code, code) for code in genre_codes]
    query_parts = [
        prompt_originale,
        " ".join(genre_codes),
        " ".join(genre_names),
        " ".join(mood_keywords),
    ]
    query_semantica = " ".join(part for part in query_parts if part).strip()
    if not query_semantica:
        query_semantica = "danza urbana coreografia ritmo movimento"

    logger.info(f"DEBUG [search_aist_dataset]: query Chroma='{query_semantica}'")

    docs = []
    try:
        docs = vector_db.similarity_search(query_semantica, k=SEARCH_K)
    except Exception as e:
        logger.error(f"[search_aist_dataset] Errore Chroma: {e}")
        payload = {
            "success": False,
            "error": f"Errore durante la ricerca Chroma: {e}",
            "risultati": [],
            "totale_trovati": 0,
        }
        return _store_search_status(payload)

    candidati_grezzi = [normalizza_metadati_candidato(dict(doc.metadata)) for doc in docs]
    risultati_filtrati = filtra_candidati_tecnici(
        candidati_grezzi,
        genre_codes=genre_codes,
        difficulty=difficulty,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    # Se il ranking semantico iniziale non porta abbastanza risultati, facciamo query più ampia
    # sempre dentro Chroma, poi applichiamo gli stessi filtri tecnici.
    n_sequenze = _get_forced_n_sequenze(n_sequenze)
    target_min = min(max(n_sequenze * 2, 10), 40)
    if len(risultati_filtrati) < target_min:
        fallback_query = " ".join(["danza coreografia", " ".join(genre_codes), " ".join(genre_names)]).strip()
        try:
            docs_fallback = vector_db.similarity_search(fallback_query or "danza coreografia", k=SEARCH_K)
            extra = [normalizza_metadati_candidato(dict(doc.metadata)) for doc in docs_fallback]
            risultati_filtrati.extend(
                filtra_candidati_tecnici(
                    extra,
                    genre_codes=genre_codes,
                    difficulty=difficulty,
                    min_bpm=min_bpm,
                    max_bpm=max_bpm,
                )
            )
        except Exception as e:
            logger.warning(f"[search_aist_dataset] Fallback Chroma non riuscito: {e}")

    risultati_finali = deduplica_per_id(risultati_filtrati)
    strict_match_count = len(risultati_finali)

    # Se la difficoltà richiesta produce pochi candidati, completiamo con stesso genere
    # ma senza filtro difficoltà. Così una richiesta da 7 mosse non fallisce solo perché
    # nel dataset ci sono meno clip "sFM" per quello stile. La UI lo segnalerà come
    # filtro rilassato, quindi non viene nascosto.
    n_sequenze = _get_forced_n_sequenze(n_sequenze)
    if difficulty and len(risultati_finali) < n_sequenze:
        try:
            docs_relaxed = vector_db.similarity_search(query_semantica or "danza coreografia", k=SEARCH_K)
            relaxed_grezzi = [normalizza_metadati_candidato(dict(doc.metadata)) for doc in docs_relaxed]
            relaxed = filtra_candidati_tecnici(
                relaxed_grezzi,
                genre_codes=genre_codes,
                difficulty=None,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
            before = len(risultati_finali)
            risultati_finali = deduplica_per_id(risultati_finali + relaxed)
            if len(risultati_finali) > before:
                filtri_rilassati.append(
                    f"difficulty:{difficulty}->qualunque_difficolta_per_raggiungere_{n_sequenze}_sequenze"
                )
                logger.info(
                    "DEBUG [search_aist_dataset]: pochi risultati con difficulty=%r (%s). "
                    "Aggiungo candidati stesso genere senza filtro difficoltà: totale=%s.",
                    difficulty,
                    before,
                    len(risultati_finali),
                )
        except Exception as e:
            logger.warning(f"[search_aist_dataset] Fallback rilassamento difficoltà non riuscito: {e}")

    salva_cache_chroma(risultati_finali)

    logger.info(f"DEBUG [search_aist_dataset]: trovati {len(risultati_finali)} risultati validi da Chroma.")

    payload = {
        "success": len(risultati_finali) > 0,
        "source": "chroma",
        "query_semantica": query_semantica,
        "filtri": {
            "genre_codes": genre_codes,
            "difficulty": difficulty,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "tempo_description": tempo_description,
            "mood_keywords": mood_keywords,
            "bpm_filter_explicit": bpm_richiesto_esplicitamente,
        },
        "filtri_rilassati": filtri_rilassati,
        "strict_match_count": strict_match_count,
        "risultati": risultati_finali[:80],
        "totale_trovati": len(risultati_finali),
        "next_tool_required": "build_sequence_plan",
    }

    if not risultati_finali:
        payload["message"] = "Nessuna sequenza trovata con questi filtri tecnici. Nota: i BPM vengono applicati solo se richiesti esplicitamente; prova ad allargare genere/difficoltà."

    return _store_search_status(payload)



# TOOL 3 — BUILD PLAN
def _normalizza_input_candidati(
    candidati_json: Optional[Union[str, dict[str, Any], list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    if candidati_json is None:
        return LAST_CHROMA_RESULTS_LIST

    if isinstance(candidati_json, str):
        try:
            dati = json.loads(candidati_json)
        except json.JSONDecodeError:
            logger.warning("build_sequence_plan: candidati_json stringa non JSON. Uso cache Chroma.")
            return LAST_CHROMA_RESULTS_LIST
    elif isinstance(candidati_json, list):
        dati = {"risultati": candidati_json}
    elif isinstance(candidati_json, dict):
        dati = candidati_json
    else:
        return LAST_CHROMA_RESULTS_LIST

    risultati = dati.get("risultati", []) if isinstance(dati, dict) else []
    if not risultati and LAST_CHROMA_RESULTS_LIST:
        logger.warning("build_sequence_plan: candidati vuoti. Uso ultima ricerca Chroma completa.")
        return LAST_CHROMA_RESULTS_LIST

    completati = [completa_candidato_da_chroma(x) for x in risultati if isinstance(x, dict)]

    # Se i candidati sono pochi/tagliati ma la cache Chroma è più ricca, preferisci la cache.
    if LAST_CHROMA_RESULTS_LIST and len(completati) < len(LAST_CHROMA_RESULTS_LIST):
        ids_completati = {x.get("id") for x in completati if x.get("id")}
        for item in LAST_CHROMA_RESULTS_LIST:
            if item.get("id") not in ids_completati:
                completati.append(item)

    return deduplica_per_id(completati)


def _estrai_json_da_testo(testo: Any) -> Any:
    """
    Estrae un oggetto JSON da una risposta LLM.
    Accetta JSON puro, blocchi ```json ... ``` o testo con JSON incorporato.
    """
    if isinstance(testo, (dict, list)):
        return testo

    if not isinstance(testo, str):
        testo = str(testo)

    testo = testo.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", testo, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        testo = fenced.group(1).strip()

    try:
        return json.loads(testo)
    except Exception:
        pass

    start_positions = [pos for pos in [testo.find("{"), testo.find("[")] if pos != -1]
    if not start_positions:
        raise ValueError("Nessun JSON trovato nella risposta del modello.")

    start = min(start_positions)
    opening = testo[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(testo)):
        ch = testo[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return json.loads(testo[start : i + 1])

    raise ValueError("JSON trovato ma non bilanciato nella risposta del modello.")


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group(0))
    return None


def _estrai_indici_da_testo_libero(
    testo: str,
    indici_validi: set[int],
    n_sequenze: int,
) -> list[int]:
    """Recupera indici anche se il modello non restituisce JSON valido.

    Esempi supportati:
    - "Scelgo gli indici 0, 2, 4, 6, 8"
    - "indice_candidato: 0, indice_candidato: 3"
    - "[0, 3, 5, 7, 9]"
    """
    testo = testo or ""
    trovati: list[int] = []

    # Prima cerca numeri vicino a parole chiave, per evitare di prendere numeri da ID o descrizioni.
    pattern_indici = re.findall(
        r"(?:indice_candidato|indice|candidate_index|indici|indices)\D{0,20}(-?\d+)",
        testo,
        flags=re.IGNORECASE,
    )

    # Se non trova abbastanza, usa tutti i numeri presenti come recupero estremo.
    numeri = pattern_indici + re.findall(r"\b-?\d+\b", testo)

    for match in numeri:
        idx = _coerce_int(match)
        if idx is None:
            continue
        if idx in indici_validi and idx not in trovati:
            trovati.append(idx)
        if len(trovati) >= n_sequenze:
            break

    return trovati


def _scelte_da_indici_recuperati(indici: list[int]) -> list[SequenzaScelta]:
    return [
        SequenzaScelta(
            posizione=pos,
            indice_candidato=idx,
            motivo="Indice recuperato dalla risposta testuale del modello quando il JSON non era valido.",
        )
        for pos, idx in enumerate(indici, start=1)
    ]


def _normalizza_output_llm(
    raw: Any,
    candidati: list[dict[str, Any]],
    n_sequenze: int,
    prompt_originale: str,
) -> tuple[list[SequenzaScelta], str, int]:
    """
    Converte diversi formati JSON possibili prodotti dal modello nello schema interno.

    Supporta sia il formato ideale:
      {"sequenze_ordinate": [{"posizione": 1, "indice_candidato": 0, "motivo": "..."}], ...}

    sia il formato prodotto nel log:
      {"coreografia": [10, 4, 7, 13, 18, 20, 15], "motivo": "..."}
    """
    id_to_index = {
        item.get("id"): i
        for i, item in enumerate(candidati)
        if isinstance(item, dict) and item.get("id")
    }

    descrizione = ""
    durata = n_sequenze * 8
    raw_items: Any = None

    if isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict):
        descrizione = (
            raw.get("descrizione_coreografia")
            or raw.get("descrizione")
            or raw.get("summary")
            or raw.get("motivo")
            or ""
        )
        durata = _coerce_int(raw.get("durata_stimata_sec") or raw.get("durata") or raw.get("duration")) or durata

        for key in (
            "sequenze_ordinate",
            "coreografia",
            "sequenze",
            "playlist",
            "scelte",
            "indici",
            "indices",
            "candidate_indices",
        ):
            if key in raw:
                raw_items = raw[key]
                break
    else:
        raise ValueError("Formato output modello non supportato.")

    if raw_items is None:
        raise ValueError("Il modello non ha restituito una lista di sequenze/indici.")

    if not isinstance(raw_items, list):
        raise ValueError("La lista di sequenze restituita dal modello non è una lista.")

    raw_ints = [_coerce_int(x) for x in raw_items if not isinstance(x, dict)]
    raw_ints = [x for x in raw_ints if x is not None]
    usa_1_based = bool(raw_ints) and 0 not in raw_ints and max(raw_ints) == len(candidati)

    scelte: list[SequenzaScelta] = []
    motivo_generale = descrizione or "Scelta coreografica del modello OpenRouter basata sui candidati Chroma."

    for pos, item in enumerate(raw_items, start=1):
        indice: Optional[int] = None
        motivo = motivo_generale
        posizione = pos

        if isinstance(item, dict):
            posizione = _coerce_int(item.get("posizione") or item.get("position") or item.get("ordine")) or pos
            indice = _coerce_int(
                item.get("indice_candidato")
                or item.get("indice")
                or item.get("index")
                or item.get("candidate_index")
                or item.get("candidato")
            )
            if indice is None and item.get("id") in id_to_index:
                indice = id_to_index[item.get("id")]
            motivo = (
                item.get("motivo")
                or item.get("reason")
                or item.get("descrizione")
                or item.get("description")
                or motivo_generale
            )
        else:
            indice = _coerce_int(item)

        if indice is None:
            continue

        if usa_1_based:
            indice -= 1

        scelte.append(
            SequenzaScelta(
                posizione=posizione,
                indice_candidato=indice,
                motivo=str(motivo),
            )
        )

    if not descrizione:
        descrizione = (
            f"Coreografia generata dal modello OpenRouter per '{prompt_originale}'. "
            f"Il modello ha scelto {len(scelte)} sequenze tra i candidati reali recuperati da Chroma."
        )

    return scelte[:n_sequenze], descrizione, int(durata)

#TOOL 3 — BUILD PLAN
@tool
def build_sequence_plan(
    prompt_originale: str,
    n_sequenze: int = DEFAULT_NUM_SEQUENCES,
    candidati_json: Optional[Union[str, dict[str, Any], list[dict[str, Any]]]] = None,
) -> str:
    """
    Fa scegliere al modello OpenRouter/Nemotron le sequenze coreografiche tra candidati Chroma reali.
    Python valida indici, completa metadati e salva playlist.json.
    """
    prompt_originale = _get_locked_prompt(prompt_originale)
    n_sequenze = _get_forced_n_sequenze(n_sequenze)
    risultati_validi = _normalizza_input_candidati(candidati_json)

    if not risultati_validi:
        error_json = json.dumps(
            {
                "success": False,
                "error": "Nessuna sequenza valida trovata. Verifica che Chroma sia stato creato e che la ricerca abbia risultati.",
                "sequenze_ordinate": [],
            },
            ensure_ascii=False,
        )
        _set_workflow_status(built=False, last_playlist_json=error_json)
        return error_json

    if len(risultati_validi) < n_sequenze:
        logger.warning(
            f"Richieste {n_sequenze} sequenze, ma disponibili solo {len(risultati_validi)} candidati validi. "
            "Ridimensiono n_sequenze."
        )
        n_sequenze = len(risultati_validi)

    numero_candidati = min(max(n_sequenze * 2, 10), MAX_CANDIDATES_FOR_LLM, len(risultati_validi))
    risultati_per_prompt = diversifica_candidati(risultati_validi, numero_candidati)

    candidati_con_indice = []
    for i, item in enumerate(risultati_per_prompt):
        item = normalizza_metadati_candidato(item)
        candidati_con_indice.append(
            {
                "indice_candidato": i,
                "indice": i,  # compatibilità con eventuali output vecchi del modello
                "id": item.get("id"),
                "genre_code": item.get("genre_code"),
                "genre_name": item.get("genre_name"),
                "bpm": item.get("bpm"),
                "difficulty": item.get("difficulty"),
                "situation_code": item.get("situation_code"),
                "music_code": item.get("music_code"),
                "dancer_id": item.get("dancer_id"),
            }
        )

    candidati_str = json.dumps(candidati_con_indice, indent=2, ensure_ascii=False)

    llm = crea_llm_tool(temperature=0, 
                        reasoning=False, 
                        num_predict=4096)

    system_msg = """
Sei un selettore di sequenze coreografiche per un workflow ReAct supervisionato.
Devi scegliere e ordinare una coreografia usando SOLO i candidati forniti.

Ogni candidato ha un campo "indice_candidato".

REGOLE OBBLIGATORIE:
- Rispondi SOLO con JSON valido.
- Non scrivere spiegazioni fuori dal JSON.
- Non usare markdown.
- Non usare blocchi ```json.
- Non inventare indici.
- Non modificare gli ID dei candidati.
- Usa ogni indice_candidato al massimo una volta.
- Ogni indice_candidato scelto deve essere presente nella lista dei candidati.
- Scegli esattamente il numero di sequenze richiesto.
- Ordina gli indici per creare una progressione: intro, sviluppo, picco, finale.
- Scrivi una breve motivazione che spieghi il criterio usato per scegliere e ordinare le sequenze.
- La motivazione deve basarsi esclusivamente sui metadati disponibili:
  genere, BPM, difficoltà, musica, ballerino e richiesta originale.
- Non affermare di avere analizzato pose, articolazioni o continuità fisica,
  perché queste informazioni non sono disponibili.
- La motivazione deve essere lunga al massimo due frasi.

FORMATO JSON OBBLIGATORIO:
{{"sequenze_ordinate":[0,1,2,3,4],"criterio_scelta":"Breve spiegazione della scelta e dell'ordine."}}

Non aggiungere campi diversi da "sequenze_ordinate" e "criterio_scelta".
Rispondi SEMPRE e SOLO con JSON valido.
"""

    user_msg = """
Richiesta originale:
{prompt}

Numero sequenze da selezionare:
{n}

Candidati disponibili:
{candidati}

Scegli e ordina la coreografia finale.
"""

    prompt_interno = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", user_msg),
        ]
    )

    logger.info(
        f"DEBUG [build_sequence_plan]: invio a OpenRouter/Nemotron {len(risultati_per_prompt)} candidati "
        f"(su {len(risultati_validi)} validi), richiesta={n_sequenze} sequenze."
    )

    scelte_llm: list[SequenzaScelta] = []
    descrizione_llm = ""
    criterio_scelta = ""
    durata_llm = n_sequenze * 8
    errore_llm: Optional[str] = None
    llm_usato = False

    try:
        messages = prompt_interno.format_messages(
            prompt=prompt_originale,
            n=n_sequenze,
            candidati=candidati_str,
        )
        risposta = llm.invoke(messages)
        contenuto = getattr(risposta, "content", risposta)
        raw_response = contenuto if isinstance(contenuto, str) else str(contenuto)

        logger.warning(
            "DEBUG [build_sequence_plan]: risposta grezza modello=%r",
            raw_response[:1500],
        )

        try:
            raw_json = _estrai_json_da_testo(raw_response)
            if  isinstance(raw_json, dict):
                    criterio_scelta = str(
                    raw_json.get("criterio_scelta", "")
                ).strip()
                    
            scelte_llm, descrizione_llm, durata_llm = _normalizza_output_llm(
                raw_json,
                risultati_per_prompt,
                n_sequenze,
                prompt_originale,
            )
        except Exception as json_error:
            indici_recuperati = _estrai_indici_da_testo_libero(
                raw_response,
                set(range(len(risultati_per_prompt))),
                n_sequenze,
            )
            if indici_recuperati:
                errore_llm = f"JSON non valido; indici recuperati da testo libero. Dettaglio: {json_error}"
                scelte_llm = _scelte_da_indici_recuperati(indici_recuperati)
                descrizione_llm = (
                    f"Coreografia generata dal modello OpenRouter per '{prompt_originale}'. "
                    "Gli indici sono stati recuperati dalla risposta testuale perché il JSON non era valido."
                )
                durata_llm = n_sequenze * 8
                logger.warning(
                    "DEBUG [build_sequence_plan]: recuperati indici da testo libero=%s",
                    indici_recuperati,
                )
            else:
                raise

        llm_usato = bool(scelte_llm)
        logger.info(f"DEBUG [build_sequence_plan]: Il modello ha scelto {len(scelte_llm)} indici validabili.")
    except Exception as e:
        errore_llm = errore_llm or str(e)
        logger.error(f"[ERRORE] Impossibile interpretare la scelta del modello OpenRouter/Nemotron: {e}")

    sequenze_finali: list[dict[str, Any]] = []
    indici_usati: set[int] = set()
    indici_validi = set(range(len(risultati_per_prompt)))
    scelte_llm_valide = 0

    if scelte_llm:
        for scelta in scelte_llm:
            indice = scelta.indice_candidato
            if indice not in indici_validi:
                logger.warning(f"[WARNING] Il modello ha scelto un indice non valido: {indice}")
                continue
            if indice in indici_usati:
                logger.warning(f"[WARNING] Il modello ha scelto un indice duplicato: {indice}")
                continue

            candidato = completa_candidato_da_chroma(risultati_per_prompt[indice])
            indici_usati.add(indice)
            scelte_llm_valide += 1
            sequenze_finali.append(
                {
                    "posizione": scelta.posizione,
                    "id": candidato.get("id"),
                    "path": candidato.get("path"),
                    "extension": candidato.get("extension"),
                    "genre_code": candidato.get("genre_code"),
                    "genre_name": candidato.get("genre_name"),
                    "genere": candidato.get("genre_name"),
                    "situation_code": candidato.get("situation_code"),
                    "difficulty": candidato.get("difficulty"),
                    "music_code": candidato.get("music_code"),
                    "dancer_id": candidato.get("dancer_id"),
                    "camera_code": candidato.get("camera_code"),
                    "sequence_code": candidato.get("sequence_code"),
                    "bpm": candidato.get("bpm"),
                    "metadata_chroma": candidato,
                    "motivo": scelta.motivo,
                }
            )

    # Se Il modello ha duplicato indici o ne ha scelti pochi, riempiamo con candidati reali non usati.
    next_pos = 1
    if sequenze_finali:
        next_pos = max(int(x.get("posizione", 0) or 0) for x in sequenze_finali) + 1

    if len(sequenze_finali) < n_sequenze:
        logger.warning(
            f"Il modello ha prodotto {len(sequenze_finali)} sequenze valide su {n_sequenze}. "
            "Completo con candidati Chroma non usati."
        )
        for idx, candidato in enumerate(risultati_per_prompt):
            if idx in indici_usati:
                continue
            candidato = completa_candidato_da_chroma(candidato)
            sequenze_finali.append(
                {
                    "posizione": next_pos,
                    "id": candidato.get("id"),
                    "path": candidato.get("path"),
                    "extension": candidato.get("extension"),
                    "genre_code": candidato.get("genre_code"),
                    "genre_name": candidato.get("genre_name"),
                    "genere": candidato.get("genre_name"),
                    "situation_code": candidato.get("situation_code"),
                    "difficulty": candidato.get("difficulty"),
                    "music_code": candidato.get("music_code"),
                    "dancer_id": candidato.get("dancer_id"),
                    "camera_code": candidato.get("camera_code"),
                    "sequence_code": candidato.get("sequence_code"),
                    "bpm": candidato.get("bpm"),
                    "metadata_chroma": candidato,
                    "motivo": "Aggiunta automaticamente per completare il numero richiesto usando un candidato reale Chroma non ancora usato.",
                }
            )
            indici_usati.add(idx)
            next_pos += 1
            if len(sequenze_finali) >= n_sequenze:
                break

    sequenze_finali = sorted(sequenze_finali, key=lambda x: int(x.get("posizione", 999) or 999))[:n_sequenze]
    for i, seq in enumerate(sequenze_finali, start=1):
        seq["posizione"] = i

    if scelte_llm_valide == n_sequenze and descrizione_llm:
        descrizione = descrizione_llm
        durata = durata_llm or len(sequenze_finali) * 8
        generata_da = "OpenRouter Nemotron"
    elif scelte_llm_valide == n_sequenze:
        descrizione = (
            f"Coreografia generata dal modello OpenRouter per: '{prompt_originale}'. "
            f"Sono state selezionate {len(sequenze_finali)} sequenze reali dal database Chroma."
        )
        durata = durata_llm or len(sequenze_finali) * 8
        generata_da = "OpenRouter Nemotron"
    elif scelte_llm_valide > 0:
        descrizione = (
            f"Coreografia generata per: '{prompt_originale}'. "
            f"Il modello ha scelto {scelte_llm_valide} sequenze valide; "
            f"Python ha completato il resto con candidati reali Chroma."
        )
        durata = durata_llm or len(sequenze_finali) * 8
        generata_da = "OpenRouter Nemotron + completamento Python"
    else:
        descrizione = (
            f"Coreografia generata per: '{prompt_originale}'. "
            f"Sono state selezionate {len(sequenze_finali)} sequenze reali dal database Chroma."
        )
        durata = len(sequenze_finali) * 8
        generata_da = "Fallback Python dopo errore OpenRouter/Nemotron"

    if criterio_scelta:
        criterio_finale = criterio_scelta
        criterio_generato_da = "OpenRouter Nemotron"
    else:
        criterio_finale = (
            "Il modello non ha fornito una motivazione testuale. "
            "Le sequenze sono state selezionate tra i candidati reali recuperati da Chroma."
        )
        criterio_generato_da = "fallback descrittivo Python"

    if 0 < scelte_llm_valide < n_sequenze:
        criterio_finale += (
            f" Nemotron ha scelto {scelte_llm_valide} sequenze; "
            "Python ha completato le posizioni mancanti con candidati Chroma non ancora utilizzati."
        )
        criterio_generato_da = "OpenRouter Nemotron + nota Python"

    elif scelte_llm_valide == 0:
        criterio_finale = (
            "Nemotron non ha prodotto una selezione valida. "
            "Python ha completato la coreografia utilizzando candidati reali recuperati da Chroma."
        )
        criterio_generato_da = "fallback Python"

    playlist = {
        "success": True,
        "sequenze_ordinate": sequenze_finali,
        "descrizione_coreografia": descrizione,
        "criterio_scelta": criterio_finale,
        "criterio_generato_da": criterio_generato_da,
        "durata_stimata_sec": int(durata),
        "prompt_originale": prompt_originale,
        "numero_sequenze_richieste": n_sequenze,
        "numero_sequenze_generate": len(sequenze_finali),
        "source": "chroma",
        "generata_da": generata_da,
        "scelte_llm_valide": scelte_llm_valide,
        "scelte_completate_da_python": max(0, len(sequenze_finali) - scelte_llm_valide),
        "fallback_interno_python": bool(errore_llm) or scelte_llm_valide < len(sequenze_finali),
    }
    if errore_llm:
        playlist["warning_llm"] = errore_llm

    percorso_playlist = Path(__file__).parent / PLAYLIST_FILENAME
    try:
        with open(percorso_playlist, "w", encoding="utf-8") as f:
            json.dump(playlist, f, indent=2, ensure_ascii=False)
        logger.info(f"[SYSTEM] playlist.json esportata con successo in: {percorso_playlist}")

    
        # ESPORTAZIONE NOMI SEQUENZE IN FILE TXT
        playlist_animazioni = Path(__file__).parent / "animation.txt"

        try:
                    # Raggruppa gli ID delle animazioni per ogni dancer_id
            ballerini_animazioni = defaultdict(list)

            for sequenza in sequenze_finali:
                dancer_id = sequenza.get("dancer_id")
                animation_id = sequenza.get("id")

                if dancer_id and animation_id:
                    ballerini_animazioni[dancer_id].append(f"{animation_id}.json")

            # Scrive ogni ballerino su una nuova riga con le sue mosse separate da spazio (o virgola)
            with open(playlist_animazioni, "w", encoding="utf-8") as f:
                for dancer_id, mosse in ballerini_animazioni.items():
                    f.write(" ".join(mosse) + "\n")

            logger.info(
                f"[SYSTEM] animation.txt esportato con successo in: "
                f"{playlist_animazioni}"
            )

        except Exception as e:
            logger.error(f"[ERRORE] Impossibile salvare animation.txt: {e}")
    except Exception as e:
        logger.error(f"[ERRORE] Impossibile salvare playlist.json: {e}")
        error_payload = {"success": False, "error": f"Errore salvataggio playlist.json: {e}"}
        error_json = json.dumps(error_payload, ensure_ascii=False)
        _set_workflow_status(built=False, last_playlist_json=error_json)
        return error_json

    playlist_json = json.dumps(playlist, indent=2, ensure_ascii=False)
    _set_workflow_status(built=True, last_playlist_json=playlist_json)
    return playlist_json


all_tools = [
    parse_dance_prompt,
    search_aist_dataset,
    build_sequence_plan,
]
