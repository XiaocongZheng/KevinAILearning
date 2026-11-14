# Kevin's AI Learning Project

A local RAG-powered study assistant for the Canadian drone pilot exams. It uses a local Chroma vector database and Ollama to answer questions based on your PDF study materials.

## Multi-Agent Study Assistant

The application, document retrieval, and chat model run locally. No Azure/Foundry or Gemini API is required. Three specialized agents work together:

- **Roadmap and Onboarding Agent**: Enter `outline` or ask “我完全不懂，第一天该学什么” to get an introduction and a modular study plan.
- **Knowledge Explainer and Translator Agent**: Ask about a concept to get a plain-language explanation, English terminology, and PDF source/page references.
- **Scenario-Based Quiz Agent**: Ask for a practice question, such as “出一道关于大风天气飞行的模拟题”. Submit your answer for feedback. If it is incorrect, the knowledge agent provides a review, and the topic is tracked as a weak area for the current session. Enter “继续” to practice again.

All three agents share the local RAG retriever and Chroma knowledge base. Study progress is kept in memory for the current run and is not saved after the application exits. Questions about airspace at real locations are answered definitively only when supported by the retrieved study materials.

## First-Time Setup on Windows

1. Install [Ollama for Windows](https://ollama.com/download).
2. In PowerShell, download and test the low-memory model:

   ```powershell
   ollama run qwen2.5:0.5b
   ```

   The model is downloaded the first time you run this command. When you finish testing, enter `/bye`.
3. Open PowerShell in the project directory, install the Python dependencies, and start the application:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
   .\.venv\Scripts\python.exe .\app.py
   ```

Ollama must be running while the application is in use. On Windows, the Ollama app usually starts its local service in the background. If the application cannot connect, open Ollama from the Start menu. Alternatively, run `ollama serve` in a separate PowerShell window and leave it open while using the application. If the service is already running, do not start a second instance.

The chat model uses a 2,048-token context window and Ollama unloads it after 30 seconds of inactivity to help free memory. This small model has limited response quality; verify exam and regulatory information against the official manuals.
