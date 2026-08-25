@echo off
title AVVIO AGENTE REACT SUPERVISIONATO OPENROUTER

cd /d "%~dp0"

echo ====================================================
echo  AVVIO AGENTE REACT SUPERVISIONATO + OPENROUTER
echo ====================================================

echo.
echo 1. Controllo Python...
py --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python Launcher "py" non trovato.
    echo Installa Python da https://www.python.org/downloads/ spuntando "Add python.exe to PATH".
    echo Poi disattiva gli alias Microsoft Store: Impostazioni ^> App ^> Alias di esecuzione app.
    pause
    exit /b 1
)

echo.
echo 2. Controllo API key OpenRouter...
if "%OPENROUTER_API_KEY%"=="" (
    echo ERRORE: OPENROUTER_API_KEY non trovata.
    echo Esegui prima questo comando in un terminale:
    echo setx OPENROUTER_API_KEY "la_tua_api_key_openrouter"
    echo Poi chiudi e riapri il terminale o riavvia questo file .bat.
    pause
    exit /b 1
)

echo.
echo 3. Creo ambiente virtuale se manca...
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 (
        echo ERRORE: impossibile creare .venv.
        pause
        exit /b 1
    )
)

echo.
echo 4. Attivo ambiente virtuale...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERRORE: impossibile attivare .venv.
    pause
    exit /b 1
)

echo.
echo 5. Installo/aggiorno dipendenze...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERRORE: installazione dipendenze fallita.
    pause
    exit /b 1
)

echo.
echo 6. Avvio Flask con OpenRouter/Nemotron...
py app.py

pause
