from __future__ import annotations

import json
import logging
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from supervised_react import run_supervised_react_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_debug.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("AgenteDanza")

app = Flask(__name__)


def sse(tipo: str, testo: Any) -> str:
    """Formatta un messaggio Server-Sent Events compatibile con il frontend."""
    return f"data: {json.dumps({'tipo': tipo, 'testo': testo}, ensure_ascii=False)}\n\n"


@app.route("/")
def interface():
    return render_template("interface.html")


@app.route("/chiedi", methods=["POST"])
def chiedi():
    dati = request.get_json(silent=True) or {}
    domanda = (dati.get("domanda") or "").strip()
    sessione_id = dati.get("sessione_id", "default_session")

    if not domanda:
        return jsonify({"errore": "Domanda mancante"}), 400

    def genera_risposta():
        logger.info(f"NUOVA SESSIONE: {sessione_id}")
        logger.info(f"DOMANDA: {domanda}")

        step_buffer: list[tuple[str, str]] = []

        def emit_step(tipo: str, testo: str) -> None:
            logger.info(f"SUPERVISION STEP: {testo}")
            step_buffer.append((tipo, testo))

        try:
            yield sse("step", "🤖 Avvio workflow ReAct supervisionato...")

            risultato = run_supervised_react_workflow(
                domanda=domanda,
                sessione_id=sessione_id,
                emit_step=emit_step,
            )

            # Invia gli step raccolti dal controller.
            for tipo, testo in step_buffer:
                yield sse(tipo, testo)

            if not risultato.get("success"):
                yield sse("errore", risultato.get("error", "Errore durante il workflow ReAct supervisionato."))
                logger.info("SESSIONE TERMINATA")
                return

            playlist = risultato.get("playlist", {})
            supervision = risultato.get("supervision", {})

            logger.info(
                "SUPERVISED REACT RESULT: success=%s generate=%s supervision=%s",
                playlist.get("success"),
                playlist.get("numero_sequenze_generate"),
                {
                    "react_completed": supervision.get("react_completed"),
                    "fallback_parse": supervision.get("fallback_parse"),
                    "fallback_search": supervision.get("fallback_search"),
                    "fallback_build": supervision.get("fallback_build"),
                    "fallback_rebuild_wrong_count": supervision.get("fallback_rebuild_wrong_count"),
                    "fallback_model_inside_build": supervision.get("fallback_model_inside_build"),
                    "fallback_filter_relaxation": supervision.get("fallback_filter_relaxation"),
                    "relaxed_filters_used": supervision.get("relaxed_filters_used"),
                    "fallback_any": supervision.get("fallback_any"),
                    "final_requested": supervision.get("final_requested"),
                    "final_generated": supervision.get("final_generated"),
                },
            )

            descrizione = playlist.get("descrizione_coreografia") or "Coreografia generata correttamente."
            numero = playlist.get("numero_sequenze_generate", len(playlist.get("sequenze_ordinate", [])))
            richieste = playlist.get("numero_sequenze_richieste") or supervision.get("final_requested") or numero

            fallback_usati = []
            if supervision.get("fallback_parse"):
                fallback_usati.append("parse")
            if supervision.get("fallback_search"):
                fallback_usati.append("search")
            if supervision.get("fallback_build"):
                fallback_usati.append("build")
            if supervision.get("fallback_rebuild_wrong_count"):
                fallback_usati.append("rigenerazione per numero sequenze errato")
            if supervision.get("fallback_model_inside_build") or playlist.get("fallback_interno_python"):
                fallback_usati.append("completamento Python interno a build_sequence_plan")
            if supervision.get("fallback_filter_relaxation"):
                fallback_usati.append("rilassamento filtri Chroma (es. difficoltà) per raggiungere il numero richiesto")

            nota_supervisione = (
                "Nessun fallback necessario: ReAct ha completato il workflow senza correzioni."
                if not fallback_usati
                else "Fallback/correzioni usati per: " + ", ".join(fallback_usati) + "."
            )

            conteggio = f"{numero} sequenze"
            if str(richieste) != str(numero):
                conteggio = f"{numero} sequenze su {richieste} richieste"

            testo_finale = (
                f"{descrizione}\n\n"
                f"✅ playlist.json generato con {conteggio} reali dal dataset Chroma.\n"
                f"🛡️ Supervisione: {nota_supervisione}"
            )
            yield sse("testo", testo_finale)

        except Exception as e:
            logger.error("Errore durante il workflow ReAct supervisionato", exc_info=(type(e), e, e.__traceback__))
            yield sse("errore", str(e))

        logger.info("SESSIONE TERMINATA")

    return Response(
        genera_risposta(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print("Server attivo su http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
