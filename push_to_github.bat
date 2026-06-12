@echo off
REM ── Push des WM-2026-Repos zu github.com/Bgjulusgb/COWORK-WM ──────────────
REM Doppelklicken oder in cmd ausfuehren. Nutzt deine lokalen Git-Credentials
REM (Git Credential Manager oeffnet ggf. den Browser-Login).
cd /d "%~dp0"

REM Defektes .git aus der Sandbox-Session entfernen (falls vorhanden)
if exist ".git" rmdir /s /q ".git"

git init -b main
git config user.name "Bgjulusgb"
git add -A
git commit -m "WM-2026 Cowork workflow - Phase 4/5 komplett: Cowork-v3-Overrides, Divergenz-Guard, Sensitivity-Check (--sensitivity), --edge-calibrated, Daten-Fixes, Schema 1.5, 109 Tests gruen"
git remote remove origin 2>nul
git remote add origin https://github.com/Bgjulusgb/COWORK-WM.git
git push -u origin main

if errorlevel 1 (
  echo.
  echo Push fehlgeschlagen. Falls das Repo schon Commits enthaelt, stattdessen:
  echo   git push -u origin main --force
  echo ausfuehren ^(ueberschreibt den Remote-Stand^).
)
pause
