@echo off
rem voice-changer-ng launcher. Extra arguments are passed to the server, e.g.
rem   start_ng.bat --model_dir D:\AI\MMVCServerSIO\model_dir --tensorrt true
cd /d %~dp0
where uv >nul 2>nul || (echo uv is required: https://docs.astral.sh/uv/getting-started/installation/ & pause & exit /b 1)
uv sync --inexact || (pause & exit /b 1)
cd server
uv run --no-sync python MMVCServerSIO.py -p 18888 --https false %*
pause
