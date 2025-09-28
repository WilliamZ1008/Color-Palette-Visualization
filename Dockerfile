FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY palette_app.py ./

EXPOSE 7860

CMD ["python", "palette_app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
