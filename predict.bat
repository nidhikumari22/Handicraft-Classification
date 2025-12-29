@echo off
python -m streamlit run predict.py
timeout /t 3 > nul
start "" http://localhost:8501