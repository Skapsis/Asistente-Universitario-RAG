"""
Servidor web mínimo para Render: responde en el puerto que exige la plataforma
para que no se mate el proceso (No open ports detected).
Se ejecuta en un hilo separado para no bloquear el bot.
"""
import os
import threading
import time

from flask import Flask

app = Flask(__name__)


@app.route("/")
def index():
    return "🤖 Bot Universitario RAG Activo"


def run_web():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


def keep_alive():
    """Abre el puerto web en un hilo para que Render detecte la app como activa."""
    t = threading.Thread(target=run_web, daemon=True)
    t.start()
    time.sleep(1)  # Dar tiempo a que el servidor enlace el puerto antes de seguir
