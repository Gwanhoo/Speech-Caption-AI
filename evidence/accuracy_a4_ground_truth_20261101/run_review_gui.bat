@echo off
setlocal
py -3.10 "%~dp0a4_review_gui.py"
if errorlevel 1 pause
