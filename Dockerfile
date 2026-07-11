FROM zeabur/claude-code:latest

USER root

RUN apt-get update && apt-get install -y python3-pip --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6" httpx==0.27.2 pydantic==2.8.2

RUN mkdir -p /agent
COPY server.py /agent/server.py
COPY CLAUDE.md /root/CLAUDE.md

WORKDIR /root

EXPOSE 8080

CMD ["python3", "/agent/server.py"]
