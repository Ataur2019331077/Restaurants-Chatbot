# Restaurant Chatbot using LangChain

This project contains two Jupyter notebooks demonstrating how to build a Restaurant Chatbot using LangChain and two different LLM backends: **OpenAI** and **Groq (LLaMA3)**.

---

## 📁 Files

- `RestuarantChatbot.ipynb`: Uses OpenAI's GPT model via `ChatOpenAI`.
- `RestuarantChatbot_usingGroq.ipynb`: Uses Groq's `llama3-8b-8192` model via `ChatGroq`.

---

## 🚀 Features

- Retrieval-Augmented Generation (RAG)
- Custom Knowledge Base for Restaurant Information
- Uses `Chroma` as Vector Store
- JinaAI / BAAI Embeddings for dense vector representation

---

## 🧠 Requirements

```bash
pip install langchain chromadb huggingface_hub groq openai python-dotenv
```

---

## 🔑 Setting API Keys

Before running, make sure to set your keys:
```python
import os

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["GROQ_API_KEY"] = "your-groq-api-key"
```

---

## 🧪 Sample Usage

```python
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-small-en",
    model_kwargs={{"trust_remote_code": True}}
)

vectorstore = Chroma.from_documents(texts, embeddings)
qa_chain = load_qa_chain(llm, chain_type="stuff")
```

---

## 📬 Output Example

> **User**: Do you have coffee?  
> **Chatbot**: Yes, Gourmet Haven has a variety of coffee options available. They offer Cold Brew Coffee, Americano Coffee, Latte Coffee, Cappuccino Coffee, and Espresso Coffee

---

## 📌 Notes

- `RestuarantChatbot.ipynb` uses `ChatOpenAI`
- `RestuarantChatbot_usingGroq.ipynb` uses `ChatGroq` (Groq API)
- Embeddings must match Chroma collection vector size (usually 768 or 1536)

---

Enjoy building AI-powered restaurant assistants! 🍝🤖

## License
This project is under [MIT License](./LICENSE).
