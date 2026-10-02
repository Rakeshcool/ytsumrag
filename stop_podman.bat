@echo off
echo [1/2] Stopping containers...
podman compose down

echo [2/2] Stopping Podman machine...
podman machine stop

echo.
echo Done.
pause