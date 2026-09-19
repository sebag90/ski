FROM docker.io/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plotter.py server.py index.html ./
COPY places/ ./places/

ENV PORT=80
ENV STATION_DISTANCE=50
EXPOSE 80

CMD ["python", "server.py"]
