@echo off
setlocal
cd /d %~dp0

REM 依你的 Anaconda 安裝路徑修改（常見兩種擇一）
REM set CONDA_EXE=C:\Users\%USERNAME%\anaconda3\Scripts\conda.exe
set CONDA_EXE=C:\ProgramData\miniconda3\Scripts\conda.exe

"%CONDA_EXE%" run -n wafflelid python detect_and_yolo.py

pause
endlocal
