# Asistente Universitario IA en Telegram (RAG)

## Descripción

Este proyecto implementa un bot de Telegram que actúa como un **tutor universitario personal**.  
Utiliza **IA generativa** y una arquitectura **RAG (Retrieval-Augmented Generation)** para responder preguntas basándose **estrictamente** en documentos PDF proporcionados por el usuario (apuntes, programas de estudio, PDFs de materias, etc.).

El bot está pensado para estudiantes universitarios que quieran:

- Hacer consultas rápidas sobre sus apuntes.
- Obtener explicaciones guiadas tipo tutor.
- Mantener conversaciones de estudio con memoria de contexto.

## Características principales

- **Carga de PDFs al vuelo**: el usuario puede enviar PDFs por Telegram y el bot:
  - Extrae el texto.
  - Lo indexa en un **vector store FAISS**.
  - Lo añade a su base de conocimiento para futuras preguntas.

- **Memoria conversacional**:
  - El bot recuerda las últimas interacciones del usuario (por chat).
  - Permite hacer **preguntas de seguimiento** sin repetir contexto.
  - Incluye un comando para **borrar la memoria** y cambiar de tema (`/limpiar`, `/reset`, `/olvidar`).

- **Prevención de alucinaciones (Prompt estricto)**:
  - El LLM está configurado como un **“Tutor Estricto”**.
  - **Solo** responde usando información presente en los PDFs cargados.
  - Si una respuesta no está en el contexto, responde:  
    > "Lo siento, esa información no está en tus apuntes."

- **Citas de fuentes (archivo y página)**:
  - Al final de cada respuesta, el bot indica:
    - El **nombre del PDF**.
    - La **página** exacta de donde extrajo la información.
  - Ejemplo:
    - `📚 Fuentes:`  
      `- Redes_I_LCIK.pdf (Pág. 4)`

- **Modelo Gemini 2.5 Flash**:
  - Usa la API de **Google Gemini** (vía `langchain-google-genai`) con el modelo:
    - `gemini-2.5-flash`
  - Optimizado para:
    - Bajas latencias.
    - Uso eficiente en entornos de despliegue gratuitos (como Render).

## Tecnologías utilizadas

- **Lenguaje**: Python
- **Framework de IA / RAG**: LangChain (`langchain`, `langchain-community`, `langchain-classic`)
- **LLM**: Google Gemini (vía `langchain-google-genai` y `google-generativeai`)
- **Vector Store**: FAISS (`faiss-cpu`) + embeddings de Hugging Face (`langchain-huggingface`, `sentence-transformers`)
- **Carga de PDFs**: `PyPDFLoader`, `PyPDFDirectoryLoader` (`pypdf`)
- **Bot de Telegram**: `pyTelegramBotAPI` (librería `telebot`)
- **Otras**:
  - `python-dotenv` para variables de entorno.
  - `requests` + `urllib3` para integraciones HTTP.
  - `pandas` para lectura de horarios desde Excel.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu_usuario>/<tu_repo>.git
cd <tu_repo>
```

### 2. Crear y activar un entorno virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Configuración de variables de entorno

El proyecto usa un archivo `.env` (que **NO** debe subirse a Git, ya está en `.gitignore`) para almacenar credenciales sensibles.

En la raíz del proyecto, crea un archivo llamado `.env` con el siguiente contenido:

```env
TELEGRAM_BOT_TOKEN=TU_TOKEN_DE_TELEGRAM_AQUI
GOOGLE_API_KEY=TU_API_KEY_DE_GOOGLE_GEMINI_AQUI
MOODLE_URL=https://grado.pol.una.py           # Opcional, si integras Moodle
MOODLE_WSTOKEN=TU_TOKEN_DE_MOODLE_SI_APLICA  # Opcional
```

- **`TELEGRAM_BOT_TOKEN`**:
  - Lo obtienes desde **@BotFather** en Telegram.
  - No lo compartas ni lo subas a repos públicos.

- **`GOOGLE_API_KEY`**:
  - Lo obtienes desde la consola de **Google AI Studio** / Google Gemini.
  - Dale permisos para usar el modelo `gemini-2.5-flash`.

## Uso

### 1. Ejecutar el bot en local

Con el entorno virtual activado y el `.env` configurado:

```bash
python bot.py
```

En la consola deberías ver algo similar a:

```text
Configurando RAG (carga de PDFs desde Fuente_Materias)...
🤖 Bot iniciado y listo en la nube...
Bot en ejecución (Long Polling). Detén con Ctrl+C.
```

### 2. Interactuar con el bot en Telegram

- Buscar tu bot en Telegram usando el nombre configurado con **@BotFather**.
- Enviar mensajes en lenguaje natural, por ejemplo:
  - "¿Qué temas entran en el primer parcial de Redes I?"
  - "Explícame el Teorema de Bayes."
  - "¿Qué es una base de datos relacional?"
- Enviar PDFs:
  - Adjunta un PDF (apuntes, programa de estudios, etc.).
  - El bot lo leerá, lo indexará y quedará disponible para futuras preguntas.

### 3. Comandos útiles

- `/start`  
  Muestra un mensaje de bienvenida y explica cómo interactuar con el asistente.

- `/limpiar`, `/reset` o `/olvidar`  
  Borra el **historial de conversación** de tu chat para que el bot “olvide” el contexto previo y puedas cambiar de tema sin confundir a la IA.

> Nota: El bot usa Long Polling (`bot.infinity_polling()`), lo cual es ideal para despliegues sencillos (ej. Render, Railway, etc.) sin necesidad de configurar Webhooks.

## Despliegue (ejemplo: Render)

1. Sube este repositorio a tu cuenta de GitHub.
2. En Render, crea un nuevo **Web Service** apuntando a tu repo.
3. Configura:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`
4. Define las variables de entorno en el panel de Render:
   - `TELEGRAM_BOT_TOKEN`
   - `GOOGLE_API_KEY`
5. Despliega el servicio. En los logs deberías ver:
   - `🤖 Bot iniciado y listo en la nube...`

## Aviso de seguridad

- Nunca subas tu archivo `.env` a GitHub (ya está excluido en `.gitignore`).
- Trata con cuidado los tokens de Telegram y las API Keys de Google.
- Si sospechas que un token se filtró, **revócalo** y genera uno nuevo.

---

Este proyecto está pensado como ejemplo de **portafolio profesional**, demostrando:

- Integración de IA generativa (Gemini) con RAG.
- Buenas prácticas de manejo de secretos.
- Arquitectura modular para futuros hooks (Moodle, horarios, etc.).

¡Siéntete libre de hacer *fork* y adaptarlo a tu carrera o universidad!

