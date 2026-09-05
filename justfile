build:
    podman build -t ski .

run: build
    podman run --rm -p 8080:80 ski
