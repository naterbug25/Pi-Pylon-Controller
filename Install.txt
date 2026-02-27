@echo off
echo   C C C C   R R R R    O O O O O  W       W       W   N     N
echo  C          R      R   O       O   W     W W     W    NN    N
echo  C          R R R R    O       O    W   W   W   W     N N   N
echo  C          R    R     O       O     W W     W W      N  N  N
echo   C C C C   R     R    O O O O O      W       W       N   N N
echo 
echo        .--.
echo       lo_o l
echo       l:_/ l
echo      //   \ \
echo     (l     l )
echo    /'\_   _/`\
echo    \___)=(___/

:: Create new Virtual env if it does not exist

python -m venv .venv
echo Create venv
call .venv\Scripts\activate.bat
echo Activate venv
::python -m pip install --upgrade pip
echo Upgrade pip
:: Install the dev tools & the project itself in editable mode
pip install -r Program/requirements.txt
echo Install requirements
:: Activate .venv
call .venv\Scripts\activate
::python Main\main.py
pause