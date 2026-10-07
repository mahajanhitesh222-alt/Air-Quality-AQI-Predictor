@echo off
cd /d "%~dp0"
echo Starting Air Quality AQI Predictor...
echo Open http://localhost:8501 in your browser
python -m streamlit run app.py --server.port 8501
