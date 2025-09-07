FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but useful for pandas performance)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY eth_bybit_notifier.py /app/eth_bybit_notifier.py

# Env vars will be injected by the platform
# ENV TG_TOKEN=__YOUR_TOKEN__ TG_CHAT_ID=__YOUR_CHAT_ID__

CMD ["python","/app/eth_bybit_notifier.py"]
