@echo off
REM ── Push zu github.com/Bgjulusgb/COWORK-WM ── idempotent ──────────────────
REM Erster Lauf: initialisiert das Repo. Jeder weitere Lauf: committet nur die
REM Aenderungen und pusht fast-forward. .git wird NIE mehr geloescht.
cd /d "%~dp0"

if not exist ".git" (
  git init -b main
  git config user.name "Bgjulusgb"
  git remote add origin https://github.com/Bgjulusgb/COWORK-WM.git
)

git add -A
git commit -m "Update %date% %time%" || echo Nichts zu committen - pushe vorhandene Commits.
git push -u origin main

if errorlevel 1 (
  echo.
  echo Push abgelehnt. EINMALIG noetig, wenn die lokale Historie neu aufgesetzt
  echo wurde ^(Remote enthaelt nur deine eigenen aelteren Pushes^):
  echo   git push -u origin main --force
)
pause
