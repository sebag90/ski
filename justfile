serve:
    python3 -m http.server 8000

build:
    podman build -t ski .

run: build
    podman run --rm -p 8080:80 ski
