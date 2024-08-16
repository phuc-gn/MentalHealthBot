FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY .env /app/.env
COPY deploy/* /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

ENTRYPOINT ["sh", "-c", "python discord_bot.py & python llm_server.py"]