from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv
from rag_indexer import get_context
from datetime import datetime, timezone
import traceback
from uuid import uuid4

# ✅ Load environment variables
load_dotenv()
app = Flask(__name__)

# --- Configurations ---
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

# ✅ Validate environment variables
if not API_KEY:
    raise EnvironmentError("Missing GROQ_API_KEY in .env file")

# --- System Prompt ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are 'Engineer AI' — a highly knowledgeable assistant for engineering, programming, and technical topics.\n"
        "Always provide accurate, relevant, and concise answers. If the answer is based on retrieved context, start the response with: '[Based on context]'.\n"
        "If the answer is based on your own knowledge, begin with: '[General knowledge]'.\n"
        "Maintain clarity with bullet points, formulas, or code snippets when useful.\n"
        "Avoid guessing or hallucinating. Say 'Not enough information' if needed.\n"
        "Be helpful, professional, and factual in tone."
    )
}
messages = [SYSTEM_PROMPT]

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        # ✅ Get relevant RAG context
        context = get_context(user_input)

        # ✅ Construct final prompt
        if context.strip():
            final_prompt = (
                f"Context:\n{context}\n\n"
                f"User Query:\n{user_input}\n\n"
                f"As Engineer AI, answer the user's query clearly using the above context.\n"
                f"- Start your answer with: [Based on context]\n"
                f"- Use bullet points, formulas, or code if helpful\n"
                f"- If the context is incomplete, say so politely and then continue with [General knowledge]."
            )
        else:
            final_prompt = (
                f"User Query:\n{user_input}\n\n"
                f"No context was found. As Engineer AI, respond using general engineering knowledge.\n"
                f"- Start your answer with: [General knowledge]\n"
                f"- Be clear and concise. Use formulas or examples where helpful.\n"
                f"- If unsure, mention that information may be limited."
            )

        messages.append({"role": "user", "content": final_prompt})

        # ✅ Make request to GROQ API
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.5
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        start_time = datetime.now(timezone.utc)
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        end_time = datetime.now(timezone.utc)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})

            # ✅ Save chat log
            with open("chat_logs.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\nUser: {user_input}\n\nReply: {reply}\n{'-'*40}")

            return jsonify({"reply": reply})
        else:
            return jsonify({
                "error": "Failed to get response from GROQ API",
                "details": response.text
            }), response.status_code

    except Exception as e:
        print("❌ Internal Server Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
