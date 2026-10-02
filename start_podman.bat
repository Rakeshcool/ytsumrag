@echo off
echo Starting Podman machine...
podman machine start

echo Starting containers...
podman compose up -d

echo.
echo Services started successfully.
pause