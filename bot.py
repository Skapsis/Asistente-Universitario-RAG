from keep_alive import keep_alive
keep_alive()

"""
Bot de Telegram - Asistente Universitario UNA (Facultad Politécnica).
Incluye: Moodle (tareas), Horarios desde Excel, RAG con Gemini (PDFs).
"""
import os
from datetime import datetime
import time
import glob

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
import telebot

# RAG (Fase 3): LangChain + Gemini + FAISS + memoria conversacional
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import ConversationalRetrievalChain

# Silenciar advertencias de SSL al conectar con Moodle (grado.pol.una.py)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MOODLE_URL = os.getenv("MOODLE_URL", "https://grado.pol.una.py").rstrip("/")
MOODLE_WSTOKEN = os.getenv("MOODLE_WSTOKEN")

# --- Rutas relativas al directorio del proyecto (compatible con Render/Linux y local) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUENTE_MATERIAS = os.path.join(BASE_DIR, "Fuente_Materias")
CARPETA_HORARIOS = os.path.join(BASE_DIR, "Fuente_Materias", "Horarios")

# --- Fase 2: Horarios desde CSV ---
# Nombres de columnas del CSV (ajusta si tu archivo usa otros nombres)
# Ejemplo típico: una columna con el día (Lunes, Martes...) y otra con hora/materia
COLUMNA_DIA = "Día"          # o "Dia", "dia", "day"
COLUMNA_HORA = "Hora"        # ej. "08:00 - 10:00"
COLUMNA_MATERIA = "Materia"  # o "Clase", "Asignatura"

# --- Fase 3: RAG (cadena global y vectorstore global, se inicializan en configurar_rag) ---
rag_chain = None
vectorstore = None

# Memoria conversacional: historial por chat_id (lista de (pregunta, respuesta))
historial_chats = {}

bot = telebot.TeleBot(TELEGRAM_TOKEN)


def obtener_tareas_moodle():
    """
    Obtiene las tareas (assignments) de Moodle con fecha de entrega futura.
    Retorna un mensaje formateado en Markdown o un mensaje de error.
    """
    if not MOODLE_WSTOKEN:
        return "❌ No está configurado `MOODLE_WSTOKEN` en el archivo `.env`."

    url = f"{MOODLE_URL}/webservice/rest/server.php"
    params = {
        "wstoken": MOODLE_WSTOKEN,
        "wsfunction": "mod_assign_get_assignments",
        "moodlewsrestformat": "json",
    }

    try:
        response = requests.post(url, data=params, verify=False, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"❌ Error al conectar con Moodle: {e}"
    except ValueError:
        return "❌ Moodle devolvió una respuesta que no es JSON válido."

    # Si la API devuelve error (ej. token inválido)
    if isinstance(data, dict) and "exception" in data:
        return f"❌ Moodle: {data.get('message', data.get('exception', 'Error desconocido'))}"

    now_ts = int(datetime.now().timestamp())
    lineas = []

    courses = data.get("courses") if isinstance(data, dict) else []
    if not courses:
        return "✅ *Todo al día*\n\nNo hay tareas pendientes con fecha de entrega futura."

    for course in courses:
        course_name = course.get("fullname", "Curso sin nombre")
        assignments = course.get("assignments") or []

        for assignment in assignments:
            duedate = assignment.get("duedate") or 0
            if duedate <= 0 or duedate <= now_ts:
                continue

            try:
                fecha_legible = datetime.fromtimestamp(duedate).strftime("%d/%m/%Y %H:%M")
            except (OSError, ValueError):
                fecha_legible = str(duedate)

            nombre_tarea = assignment.get("name", "Tarea sin nombre")
            lineas.append(f"• *{nombre_tarea}*\n  📚 {course_name}\n  📅 Vence: {fecha_legible}")

    if not lineas:
        return "✅ *Todo al día*\n\nNo hay tareas pendientes con fecha de entrega futura."

    titulo = "📋 *Tareas pendientes (Moodle)*\n"
    return titulo + "\n\n".join(lineas)


def configurar_rag():
    """
    Carga todos los PDFs de Fuente_Materias con PyPDFDirectoryLoader,
    crea el vectorstore con FAISS y lo deja en la variable global vectorstore,
    y la cadena RAG en la variable global rag_chain.
    """
    global rag_chain, vectorstore
    try:
        print("Cargando documentos...")
        loader = PyPDFDirectoryLoader(FUENTE_MATERIAS)
        docs = loader.load()
        if not docs or len(docs) == 0:
            print("⚠️ RAG: La carpeta Fuente_Materias está vacía o no tiene PDFs.")
            return None

        print("Dividiendo texto en fragmentos...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        if not splits or len(splits) == 0:
            print("⚠️ No se encontró texto seleccionable en los PDFs de Fuente_Materias.")
            return None

        print("Creando VectorStore...")
        # Embeddings locales con HuggingFace para evitar errores 404 de Google.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(
            splits, embeddings
        )  # global para poder añadir PDFs desde el manejador de documentos

        # max_retries para manejar micro-cortes de red con la API de Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=3)

        # Prompt de Tutor Proactivo: responde SOLO con el contexto de los PDFs,
        # pero de forma activa, guiando al estudiante y proponiendo siguientes pasos.
        # Además, si el contexto incluye documentos tipo "Programa de Estudios" o índices,
        # el tutor debe usar esa información solo como referencia de secciones y luego buscar
        # explicaciones teóricas detalladas en los demás archivos disponibles.
        plantilla = """Eres un Tutor Universitario Experto y Proactivo. Tienes dos fuentes de información: el Contexto (apuntes del usuario) y tu propio conocimiento experto.
REGLA 1: Siempre revisa primero el Contexto proporcionado.
REGLA 2: Si el Contexto contiene la teoría y explicación completa, úsala para responder e indica tus fuentes.
REGLA 3 (CRÍTICA): Si el Contexto SOLO contiene un programa de estudios, un índice o una lista de temas (por ejemplo, "Unidad 2", "2.1 Tema A", "2.2 Tema B") sin desarrollo teórico, NO te detengas. Extrae esos temas del contexto y USA TU PROPIO CONOCIMIENTO EXPERTO para explicarlos a fondo, como si estuvieras dando la clase.
REGLA 4: Cuando uses tu propio conocimiento para rellenar los vacíos del programa de estudios, inicia tu respuesta con esta frase exacta: "💡 Tus apuntes solo listan los temas de esta unidad, así que he investigado la teoría detallada para ti:" y luego desarrolla la clase de forma estructurada.
IMPORTANTE: Los metadatos o el nombre del archivo fuente (por ejemplo, Administracion-IV.pdf) te indican de qué materia trata el texto. Si el usuario pregunta por "Administración IV", asume que cualquier contexto proveniente de ese archivo PERTENECE a esa materia, aunque el texto del párrafo no mencione explícitamente el nombre de la asignatura.
Sé flexible con los números: "Unidad 2" es lo mismo que "Unidad II" o "Capítulo 2".
Si el usuario te pide hablar sobre una unidad basada en una tarea pendiente, busca los temas principales de esa unidad en el contexto y explícalos.
Si el usuario pide más información o dice cosas como "dime más", "ampliar" o "explica mejor", busca en el contexto información más técnica, ejemplos prácticos o comparaciones que no hayas mencionado antes.
Si el contexto menciona solo un programa de estudios, índice o lista de temas, usa esa información para identificar la sección o unidad, pero cuando no haya desarrollo teórico suficiente, aplica la REGLA 3 y usa tu propio conocimiento experto para desarrollar la teoría.
Si el usuario pregunta por una sección o unidad, busca la explicación teórica en todos los archivos disponibles. No te limites a decir que el tema existe, ¡explícalo!
Al finalizar cada respuesta, ofrece 2 o 3 puntos clave o subtemas de la sección actual para seguir profundizando.
Si el usuario pregunta por horarios o exámenes, busca en los documentos que mencionen "Horario", "Calendario" o "Cronograma". Usa la fecha actual proporcionada para filtrar la información relevante (por ejemplo, si hoy es lunes, prioriza las clases de los lunes).
Contexto: {context}
Pregunta: {question}
Respuesta:"""
        QA_PROMPT = PromptTemplate(template=plantilla, input_variables=["context", "question"])

        # Prompt para condensar/reformular la pregunta usando el historial y evitar perder contexto
        plantilla_condense = """Dado el siguiente historial de conversación entre un estudiante y su tutor universitario, y la nueva pregunta del estudiante, reformula la pregunta para que sea independiente, específica y adecuada para buscar en los apuntes.

Historial:
{chat_history}

Nueva pregunta del estudiante:
{question}

Instrucciones:
- Si la nueva pregunta es ambigua o del tipo "dime más", "seguí contándome", "explica mejor", etc., reemplázala por una pregunta completa que deje claro el tema, por ejemplo: "detalles adicionales sobre [tema anterior]".
- NO respondas la pregunta, SOLO reformúlala como una búsqueda concreta y completa.

Pregunta reformulada:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate(
            template=plantilla_condense, input_variables=["chat_history", "question"]
        )

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20}
            ),
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )
        print("✅ RAG configurado correctamente (Gemini + FAISS + memoria conversacional).")
    except Exception as e:
        print(f"⚠️ RAG no pudo configurarse: {e}")


@bot.message_handler(content_types=["document"])
def manejar_documento(message):
    """
    Recibe un PDF por Telegram, lo descarga, extrae el texto y lo añade al vectorstore RAG
    para que el usuario pueda hacer preguntas sobre ese documento.
    """
    doc = message.document
    if not doc or doc.mime_type != "application/pdf":
        bot.reply_to(message, "Lo siento, por ahora solo puedo leer archivos PDF. 📄")
        return

    if vectorstore is None:
        bot.reply_to(
            message,
            "❌ El asistente de materiales aún no está listo. Revisa que Fuente_Materias tenga PDFs y arranca el bot de nuevo.",
        )
        return

    msg = bot.reply_to(message, "📥 Descargando e inyectando documento en mi memoria...")
    bot.send_chat_action(message.chat.id, "upload_document")
    temp_path = f"temp_{message.chat.id}.pdf"

    try:
        file_info = bot.get_file(doc.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(temp_path, "wb") as f:
            f.write(downloaded_file)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        if not docs or all(not (d.page_content or "").strip() for d in docs):
            raise ValueError(
                "Este PDF parece ser una imagen escaneada sin texto. "
                "Pásalo por un OCR primero o envía un PDF con texto seleccionable."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos_divididos = splitter.split_documents(docs)
        if not textos_divididos:
            raise ValueError(
                "No se pudo extraer texto útil. El PDF podría ser solo imágenes."
            )

        vectorstore.add_documents(textos_divididos)
        bot.edit_message_text(
            "✅ ¡Documento leído y memorizado! Ya puedes hacerme preguntas sobre este archivo.",
            chat_id=message.chat.id,
            message_id=msg.message_id,
        )
    except ValueError as e:
        bot.edit_message_text(
            f"❌ {e}",
            chat_id=message.chat.id,
            message_id=msg.message_id,
        )
    except Exception as e:
        bot.edit_message_text(
            f"❌ No se pudo procesar el PDF: {e}",
            chat_id=message.chat.id,
            message_id=msg.message_id,
        )
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def obtener_clases_hoy():
    """
    Lee el horario desde un CSV de horarios y devuelve las clases del día actual.
    Si es fin de semana, no hay archivo o no hay clases, devuelve mensaje de descanso o error.
    """
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    hoy = datetime.now().weekday()  # 0 = Lunes, 6 = Domingo

    if hoy >= 5:
        return "¡Hoy no tienes clases, a descansar! 🎮"

    nombre_dia = dias[hoy]

    try:
        # Buscar dinámicamente cualquier CSV dentro de la carpeta de horarios
        patrones = [
            os.path.join(CARPETA_HORARIOS, "*.csv"),
            os.path.join(CARPETA_HORARIOS, "**", "*.csv"),
        ]
        archivos_csv = []
        for patron in patrones:
            archivos_csv.extend(glob.glob(patron, recursive=True))

        if not archivos_csv:
            return (
                "❌ No se encontró ningún archivo de horario en la carpeta de horarios. "
                "Asegúrate de que exista al menos un archivo .csv en la carpeta de horarios."
            )

        ruta_csv = archivos_csv[0]

        df = pd.read_csv(ruta_csv)

        if df.empty:
            return "❌ El archivo de horario está vacío."

        # Buscar columna del día (por si tiene otro nombre)
        col_dia = None
        for c in [COLUMNA_DIA, "Dia", "dia", "Día"]:
            if c in df.columns:
                col_dia = c
                break
        if col_dia is None:
            return "❌ En el Excel no hay una columna de día. Ajusta `COLUMNA_DIA` en bot.py (ej. 'Día', 'Dia')."

        # Filtrar filas del día actual (normalizar para comparar)
        df[col_dia] = df[col_dia].astype(str).str.strip()
        filas_hoy = df[df[col_dia].str.lower() == nombre_dia.lower()]

        if filas_hoy.empty:
            return f"¡Hoy no tienes clases, a descansar! 🎮"

        # Construir líneas: intentar Hora + Materia, o la primera columna que tenga contenido
        col_hora = COLUMNA_HORA if COLUMNA_HORA in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        col_mat = COLUMNA_MATERIA if COLUMNA_MATERIA in df.columns else (df.columns[2] if len(df.columns) > 2 else col_hora)
        lineas = []
        for _, row in filas_hoy.iterrows():
            hora = str(row.get(col_hora, "")).strip()
            materia = str(row.get(col_mat, "")).strip()
            if hora and materia:
                lineas.append(f"• {hora}: {materia}")
            elif hora:
                lineas.append(f"• {hora}")
            else:
                lineas.append(f"• {materia}" if materia else "• —")

        titulo = f"📅 *Clases del día — {nombre_dia}*\n\n"
        return titulo + "\n".join(lineas) if lineas else "¡Hoy no tienes clases, a descansar! 🎮"

    except Exception as e:
        return f"❌ Error al leer el horario: {e}. Revisa que el Excel exista y tenga el formato esperado (columnas Día, Hora, Materia)."


def enviar_mensaje_largo(chat_id, texto):
    """
    Envía un texto que puede superar el límite de 4096 caracteres de Telegram,
    dividiéndolo en fragmentos y enviando cada uno en un mensaje secuencial.
    """
    MAX_LENGTH = 4000
    if not texto or len(texto) <= MAX_LENGTH:
        if texto:
            try:
                bot.send_message(chat_id, texto, parse_mode="Markdown")
            except Exception:
                bot.send_message(chat_id, texto)
        return
    pos = 0
    while pos < len(texto):
        fragmento = texto[pos : pos + MAX_LENGTH]
        pos += MAX_LENGTH
        try:
            bot.send_message(chat_id, fragmento, parse_mode="Markdown")
        except Exception:
            bot.send_message(chat_id, fragmento)


# -------- MANEJADORES DE TELEGRAM --------

@bot.message_handler(commands=["start"])
def comando_start(message):
    """Solo se usa la primera vez: mensaje de bienvenida."""
    texto = (
        "👋 *¡Hola!* Soy tu asistente universitario.\n\n"
        "Puedes escribirme en lenguaje natural, por ejemplo:\n"
        "• \"¿Qué tareas tengo pendientes?\"\n"
        "• \"¿Qué clases tocan hoy?\"\n"
        "• \"Hola\" o \"¿Qué tal?\"\n"
        "• Cualquier pregunta sobre tus materiales de la carrera\n"
    )
    bot.reply_to(message, texto, parse_mode="Markdown")


@bot.message_handler(commands=["limpiar", "reset", "olvidar"])
def comando_limpiar_memoria(message):
    """Borra el historial de conversación del usuario para que la IA no se confunda con el tema anterior."""
    global historial_chats
    historial_chats[message.chat.id] = []
    bot.reply_to(
        message,
        "🧹 ¡Memoria borrada exitosamente! He olvidado nuestra conversación anterior. Mi mente está en blanco. ¿De qué nuevo tema quieres que hablemos?",
    )


@bot.message_handler(func=lambda message: message.text is not None and bool(message.text.strip()))
def manejador_maestro(message):
    """
    Enrutador de intenciones: interpreta el mensaje en lenguaje natural
    y responde con tareas, horario, saludo o RAG. Todas las respuestas
    editan un mensaje temporal para una experiencia fluida.
    """
    texto_original = message.text.strip()
    mensaje = texto_original.lower()

    # Intención: Tareas (Moodle)
    if any(palabra in mensaje for palabra in ("tarea", "tareas", "moodle", "pendientes")):
        msg = bot.reply_to(message, "⏳ Revisando Moodle...")
        resultado = obtener_tareas_moodle()
        _editar_con_markdown(msg, resultado, message.chat.id)

    # Intención: Horario / clases de hoy
    elif any(palabra in mensaje for palabra in ("clase", "clases", "horario", "toca hoy")):
        msg = bot.reply_to(message, "📅 Revisando tu horario...")
        resultado = obtener_clases_hoy()
        _editar_con_markdown(msg, resultado, message.chat.id)

    # Intención: Saludo
    elif any(palabra in mensaje for palabra in ("hola", "buenas", "qué tal", "que tal", "buen día", "buenos días")):
        msg = bot.reply_to(message, "👋")
        respuesta = (
            "¡Hola! 👋 Soy tu asistente universitario. Puedo ayudarte a ver tus *tareas* en Moodle, "
            "tu *horario* del día o responder preguntas sobre los materiales de tu carrera. ¿En qué te ayudo?"
        )
        _editar_con_markdown(msg, respuesta, message.chat.id)

    # Intención: Pregunta sobre la carrera (RAG con memoria conversacional)
    else:
        if rag_chain is None:
            bot.reply_to(
                message,
                "❌ El asistente de materiales no está disponible. Revisa que Fuente_Materias tenga PDFs y GOOGLE_API_KEY en .env.",
            )
            return
        # Conocimiento del tiempo actual para preguntas relativas a "hoy", "mañana", etc.
        ahora = datetime.now()
        contexto_tiempo = ahora.strftime("%Y-%m-%d %H:%M")
        pregunta_con_tiempo = f"[Fecha y hora actual: {contexto_tiempo}] {texto_original}"

        msg = bot.reply_to(message, "🧠 Consultando tus documentos...")
        bot.send_chat_action(message.chat.id, "typing")

        # Reintentos para manejar errores temporales de red/SSL con Gemini
        intentos_max = 3
        ultimo_error = None

        for intento in range(1, intentos_max + 1):
            try:
                historial = historial_chats.get(message.chat.id, [])
                # ConversationalRetrievalChain (usar versión classic) acepta chat_history
                # como lista de tuplas (humano, bot) o lista de mensajes.
                # Aquí usamos la forma de tuplas para que se vea claramente quién dijo qué.
                chat_history_tuplas = [(preg, resp) for preg, resp in historial]

                resultado = rag_chain.invoke({"question": pregunta_con_tiempo, "chat_history": chat_history_tuplas})
                respuesta_original = resultado.get("answer", str(resultado))

                # Construir citas de fuentes (PDF + página) sin repetir la misma página
                documentos = resultado.get("source_documents", [])
                fuentes_unicas = set()
                for doc in documentos:
                    fuente = doc.metadata.get("source", "Desconocido")
                    nombre_archivo = os.path.basename(fuente)
                    pagina = doc.metadata.get("page", 0)
                    if isinstance(pagina, (int, float)):
                        numero_pag = int(pagina) + 1
                    else:
                        numero_pag = 1
                    fuentes_unicas.add((nombre_archivo, numero_pag))
                lineas_fuentes = sorted(fuentes_unicas, key=lambda x: (x[0], x[1]))
                if lineas_fuentes:
                    texto_fuentes = "\n\n📚 *Fuentes:*\n" + "\n".join(f"- {nom} (Pág. {n})" for nom, n in lineas_fuentes)
                    mensaje_al_usuario = respuesta_original + texto_fuentes
                else:
                    mensaje_al_usuario = respuesta_original

                # Si la respuesta supera el límite de Telegram (4096), dividir en varios mensajes
                if len(mensaje_al_usuario) <= 4000:
                    _editar_con_markdown(msg, mensaje_al_usuario, message.chat.id)
                else:
                    try:
                        bot.edit_message_text(
                            "✅ Respuesta a continuación:",
                            chat_id=message.chat.id,
                            message_id=msg.message_id,
                        )
                    except Exception:
                        pass
                    enviar_mensaje_largo(message.chat.id, mensaje_al_usuario)
                # Actualizar historial solo con la respuesta original (sin fuentes) para no ensuciar la memoria.
                # No limitamos la longitud del historial aquí para que la IA pueda mantener contexto amplio;
                # el usuario puede limpiarlo en cualquier momento con /limpiar, /reset u /olvidar.
                historial.append((texto_original, respuesta_original))
                historial_chats[message.chat.id] = historial
                ultimo_error = None
                break
            except Exception as e:
                ultimo_error = e
                if intento < intentos_max:
                    time.sleep(1)
                else:
                    bot.edit_message_text(
                        f"❌ Error al consultar tras varios intentos: {e}",
                        chat_id=message.chat.id,
                        message_id=msg.message_id,
                    )


def _editar_con_markdown(msg, texto, chat_id):
    """Intenta editar el mensaje con Markdown; si falla, edita sin formato."""
    try:
        bot.edit_message_text(texto, chat_id=chat_id, message_id=msg.message_id, parse_mode="Markdown")
    except Exception:
        bot.edit_message_text(texto, chat_id=chat_id, message_id=msg.message_id)


if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("❌ Falta TELEGRAM_BOT_TOKEN en el archivo .env")
    else:
        print("Configurando RAG (carga de PDFs desde Fuente_Materias)...")
        configurar_rag()
        print("🤖 Bot iniciado y listo en la nube...")
        print("Iniciando polling de Telegram...")
        try:
            bot.infinity_polling()
        except Exception as e:
            print(f"Error crítico en el bot: {e}")
