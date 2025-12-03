FROM python:3.11-slim

WORKDIR /app

# Copia arquivos
COPY requirements.txt .
COPY . .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta HF Spaces
EXPOSE 7860

# Roda Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]