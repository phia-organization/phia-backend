# 🚀 API - Project Setup & Run Guide

This is an API built using **FastAPI**. Follow the steps below to get it up and running.

---

## 📁 Required Structure

Before starting the API, make sure all models and weights are placed in the following folder:

```
models/
```

---

## ⚙️ Environment Setup

1. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. **Install project dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚦 Running the API

Start the development server with:

```bash
fastapi dev main.py
```

> 💡 Make sure the [FastAPI CLI](https://fastapi.tiangolo.com/) is installed. If not, you can install it with:
>
> ```bash
> pip install fastapi[all]
> ```

---

## ✅ Access the API

Once running, the API will be available at:

```
http://localhost:8000
```

Interactive documentation:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧠 Requirements

- Python 3.8 or higher
- FastAPI
- All dependencies listed in [`requirements.txt`](./requirements.txt)
