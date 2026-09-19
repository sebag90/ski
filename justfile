build:
    podman build -t ski .

run: build
    podman run --rm -p 8080:80 -e STATION_DISTANCE=${STATION_DISTANCE:-50} ski

dev:
    python3 server.py
