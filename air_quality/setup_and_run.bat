@echo off
echo ============================================
echo  Air Quality AQI Predictor — Setup Script
echo ============================================
echo.

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/4] Generating dataset (5000 rows)...
python generate_dataset.py
if errorlevel 1 (
    echo ERROR: Dataset generation failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Training XGBoost model...
python train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Launching Streamlit app...
echo       Open http://localhost:8501 in your browser
echo.
streamlit run app.py
