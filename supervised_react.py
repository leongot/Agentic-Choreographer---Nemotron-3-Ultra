"""
supervised_react.py — controller ReAct supervisionato.

Obiettivo:
- usare un agente ReAct/LangGraph per il tool use;
- impedire che ReAct libero rompa il workflow obbligatorio;
- garantire la sequenza: parse_dance_prompt -> search_aist_dataset -> build_sequence_plan;
- recuperare automaticamente gli step mancanti;
- preferire sempre l'ultima ricerca Chroma valida, non l'ultima ricerca in assoluto.

Perché serve:
ReAct può fermarsi dopo la search, inventare una seconda richiesta o sovrascrivere una
ricerca buona con una ricerca fallita. Questo controller mantiene l'approccio ReAct,
ma aggiunge guardrail.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

StepCallback = Optional[Callable[[str, str], None]]


def parse_json_sicuro(value: Any, fallback: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if fallback is None:
        fallback = {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            return fallback
    return fallback


def invoke_tool(tool: Any, payload: dict[str, Any]) -> Any:
    """Supporta sia LangChain tool (.invoke) sia funzioni normali, utile anche nei test."""
    if hasattr(tool, "invoke"):
        return tool.invoke(payload)
    return tool(**payload)


def _load_runtime_dependencies(
    react_agent: Any,
    parse_tool: Any,
    search_tool: Any,
    build_tool: Any,
    reset_status: Any,
    get_status: Any,
):
    """
    Import lazy: il modulo resta testabile anche senza LangChain/OpenRouter.
    In produzione importa i tool reali solo quando serve.
    """
    if parse_tool is None or search_tool is None or build_tool is None or reset_status is None or get_status is None:
        from tools import (  # type: ignore
            build_sequence_plan,
            get_workflow_status,
            parse_dance_prompt,
            reset_workflow_status,
            search_aist_dataset,
        )

        parse_tool = parse_tool or parse_dance_prompt
        search_tool = search_tool or search_aist_dataset
        build_tool = build_tool or build_sequence_plan
        reset_status = reset_status or reset_workflow_status
        get_status = get_status or get_workflow_status

    if react_agent is None:
        from agent import agent  # type: ignore

        react_agent = agent

    return react_agent, parse_tool, search_tool, build_tool, reset_status, get_status


def _safe_status(get_status: Any) -> dict[str, Any]:
    try:
        status = get_status()
        return status if isinstance(status, dict) else {}
    except Exception as e:
        logger.warning("Impossibile leggere lo stato workflow: %s", e)
        return {}


def _best_search_json(status: dict[str, Any]) -> Optional[str]:
    """Preferisce l'ultima ricerca valida, non l'ultima ricerca in assoluto."""
    successful = status.get("last_successful_search_json")
    if successful:
        return successful

    last = status.get("last_search_json")
    if last:
        parsed = parse_json_sicuro(last)
        if parsed.get("success") and int(parsed.get("totale_trovati") or 0) > 0:
            return last
    return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _playlist_generated_count(playlist: dict[str, Any]) -> int:
    explicit = _int_or_none(playlist.get("numero_sequenze_generate"))
    if explicit is not None:
        return explicit
    seqs = playlist.get("sequenze_ordinate") or []
    return len(seqs) if isinstance(seqs, list) else 0


def _playlist_requested_count(playlist: dict[str, Any], expected: int) -> int:
    explicit = _int_or_none(playlist.get("numero_sequenze_richieste"))
    return explicit if explicit is not None else expected


def _detect_internal_build_fallback(playlist: dict[str, Any]) -> bool:
    completed = _int_or_none(playlist.get("scelte_completate_da_python")) or 0
    generata_da = str(playlist.get("generata_da") or "")
    return (
        bool(playlist.get("fallback_interno_python"))
        or completed > 0
        or bool(playlist.get("warning_llm"))
        or "Fallback Python" in generata_da
        or "completamento Python" in generata_da
    )


def _invoke_react_agent(react_agent: Any, domanda: str, sessione_id: str) -> None:
    """Invoca ReAct in modo sincrono per evitare problemi di event loop su Windows/Flask."""
    payload = {"messages": [("user", domanda)]}
    config = {"configurable": {"thread_id": sessione_id}, "recursion_limit": 16}

    if hasattr(react_agent, "invoke"):
        react_agent.invoke(payload, config=config)
    else:
        react_agent(payload, config=config)


def run_supervised_react_workflow(
    domanda: str,
    sessione_id: str = "default_session",
    *,
    react_agent: Any = None,
    parse_tool: Any = None,
    search_tool: Any = None,
    build_tool: Any = None,
    reset_status: Any = None,
    get_status: Any = None,
    emit_step: StepCallback = None,
) -> dict[str, Any]:
    """
    Esegue un workflow ReAct supervisionato.

    Ritorna:
    {
      "success": bool,
      "playlist": dict,
      "playlist_json": str,
      "supervision": {...}
    }
    """
    domanda = (domanda or "").strip()
    if not domanda:
        return {
            "success": False,
            "error": "Domanda mancante.",
            "playlist": {},
            "playlist_json": "{}",
            "supervision": {"mode": "react_supervised"},
        }

    react_agent, parse_tool, search_tool, build_tool, reset_status, get_status = _load_runtime_dependencies(
        react_agent,
        parse_tool,
        search_tool,
        build_tool,
        reset_status,
        get_status,
    )

    reset_status(domanda)

    supervision: dict[str, Any] = {
        "mode": "react_supervised",
        "react_completed": False,
        "react_error": None,
        "fallback_parse": False,
        "fallback_search": False,
        "fallback_build": False,
        "fallback_rebuild_wrong_count": False,
        "fallback_model_inside_build": False,
        "fallback_filter_relaxation": False,
        "relaxed_filters_used": [],
        "fallback_any": False,
        "used_successful_search_cache": False,
        "ignored_react_prompts": [],
        "final_requested": None,
        "final_generated": None,
    }

    # 1) ReAct prova a ragionare e usare i tool.
    try:
        if emit_step:
            emit_step("step", "🤖 Avvio agente ReAct supervisionato...")
        _invoke_react_agent(react_agent, domanda, sessione_id)
        supervision["react_completed"] = True
    except Exception as e:
        supervision["react_error"] = str(e)
        logger.warning(
            "ReAct non ha completato la sessione. Il controller userà fallback controllato.",
            exc_info=(type(e), e, e.__traceback__),
        )

    status = _safe_status(get_status)
    supervision["ignored_react_prompts"] = list(status.get("ignored_react_prompts") or [])

    # 2) Garantisce parse.
    parsed_json = status.get("last_parse_json")
    if not parsed_json:
        supervision["fallback_parse"] = True
        if emit_step:
            emit_step("step", "🔍 Fallback controllato: interpreto la richiesta originale...")
        parsed_json = invoke_tool(parse_tool, {"prompt": domanda})
        status = _safe_status(get_status)
        supervision["ignored_react_prompts"] = list(status.get("ignored_react_prompts") or [])

    parsed = parse_json_sicuro(parsed_json)
    if not parsed:
        return {
            "success": False,
            "error": "Impossibile interpretare la richiesta dell'utente.",
            "playlist": {},
            "playlist_json": "{}",
            "supervision": supervision,
        }

    status_n = _int_or_none(status.get("original_n_sequenze"))
    parsed_n = _int_or_none(parsed.get("n_sequenze"))
    n_sequenze = status_n or parsed_n or 5

    # 3) Garantisce search valida. Preferisce la cache della ricerca valida.
    search_json = _best_search_json(status)
    if search_json:
        supervision["used_successful_search_cache"] = bool(status.get("last_successful_search_json"))

    if not search_json:
        supervision["fallback_search"] = True
        if emit_step:
            emit_step("step", "📂 Fallback controllato: cerco in Chroma usando il prompt originale...")
        search_json = invoke_tool(
            search_tool,
            {
                "genre_codes": parsed.get("genre_codes", []),
                "mood_keywords": parsed.get("mood_keywords", []),
                "difficulty": parsed.get("difficulty"),
                "min_bpm": parsed.get("min_bpm"),
                "max_bpm": parsed.get("max_bpm"),
                "tempo_description": parsed.get("tempo_description"),
                "prompt_originale": domanda,
                "n_sequenze": n_sequenze,
            },
        )
        status = _safe_status(get_status)
        supervision["ignored_react_prompts"] = list(status.get("ignored_react_prompts") or [])
        search_json = _best_search_json(status) or search_json

    search_result = parse_json_sicuro(search_json)
    if not search_result.get("success"):
        message = search_result.get("message") or search_result.get("error") or "Nessuna sequenza trovata."
        return {
            "success": False,
            "error": message,
            "playlist": {},
            "playlist_json": "{}",
            "supervision": supervision,
        }

    # 4) Garantisce build. Se ReAct non l'ha chiamata, la chiama il controller.
    playlist_json = status.get("last_playlist_json") if status.get("built") else None

    def _build_with_controller(reason: str) -> Any:
        supervision["fallback_build"] = True
        if reason == "wrong_count":
            supervision["fallback_rebuild_wrong_count"] = True
        if emit_step:
            msg = "🎭 Fallback controllato: genero playlist.json dai candidati Chroma validi..."
            if reason == "wrong_count":
                msg = "🎭 Fallback controllato: rigenero playlist.json perché il numero di sequenze non coincide..."
            emit_step("step", msg)
        return invoke_tool(
            build_tool,
            {
                "candidati_json": search_json,
                "prompt_originale": domanda,
                "n_sequenze": n_sequenze,
            },
        )

    if not playlist_json:
        playlist_json = _build_with_controller("missing")
        status = _safe_status(get_status)
        supervision["ignored_react_prompts"] = list(status.get("ignored_react_prompts") or [])

    playlist = parse_json_sicuro(playlist_json)

    # 5) Non basta che build_sequence_plan sia stato chiamato: deve rispettare
    # l'intento originale. ReAct può chiamare il tool corretto passando n_sequenze
    # sbagliato; se il conteggio non coincide, rigeneriamo in modo controllato.
    generated_int = _playlist_generated_count(playlist)
    requested_int = _playlist_requested_count(playlist, n_sequenze)

    if playlist.get("success", True) and (generated_int != n_sequenze or requested_int != n_sequenze):
        logger.warning(
            "Playlist generata con numero sequenze non coerente: generate=%s, requested_in_playlist=%s, attese=%s. Rigenero.",
            generated_int,
            requested_int,
            n_sequenze,
        )
        playlist_json = _build_with_controller("wrong_count")
        status = _safe_status(get_status)
        supervision["ignored_react_prompts"] = list(status.get("ignored_react_prompts") or [])
        playlist = parse_json_sicuro(playlist_json)
        generated_int = _playlist_generated_count(playlist)
        requested_int = _playlist_requested_count(playlist, n_sequenze)

    final_status = _safe_status(get_status)
    relaxed_filters_used = list(final_status.get("relaxed_filters_used") or [])
    supervision["fallback_filter_relaxation"] = bool(relaxed_filters_used)
    supervision["relaxed_filters_used"] = relaxed_filters_used
    supervision["fallback_model_inside_build"] = _detect_internal_build_fallback(playlist)
    supervision["fallback_any"] = any([
        supervision.get("fallback_parse"),
        supervision.get("fallback_search"),
        supervision.get("fallback_build"),
        supervision.get("fallback_rebuild_wrong_count"),
        supervision.get("fallback_model_inside_build"),
        supervision.get("fallback_filter_relaxation"),
    ])
    supervision["final_requested"] = requested_int
    supervision["final_generated"] = generated_int

    if not playlist.get("success", True):
        return {
            "success": False,
            "error": playlist.get("error", "Errore durante la creazione della playlist."),
            "playlist": playlist,
            "playlist_json": playlist_json if isinstance(playlist_json, str) else json.dumps(playlist_json, ensure_ascii=False),
            "supervision": supervision,
        }

    # Se dopo la rigenerazione il numero resta sbagliato non va nascosto come successo normale.
    if generated_int != n_sequenze:
        return {
            "success": False,
            "error": (
                f"La playlist generata non rispetta il numero richiesto: "
                f"richieste {n_sequenze}, generate {generated_int}."
            ),
            "playlist": playlist,
            "playlist_json": playlist_json if isinstance(playlist_json, str) else json.dumps(playlist_json, ensure_ascii=False),
            "supervision": supervision,
        }

    supervision["final_status"] = final_status
    return {
        "success": True,
        "playlist": playlist,
        "playlist_json": playlist_json if isinstance(playlist_json, str) else json.dumps(playlist_json, ensure_ascii=False),
        "supervision": supervision,
    }
