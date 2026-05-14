FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Verify artifacts are present during build
RUN ls -R /app/artifacts/

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
