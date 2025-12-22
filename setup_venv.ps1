# PowerShell script to set up virtual environment for the 3D simulation project

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv .venv

Write-Host "`nActivating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

Write-Host "`nUpgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "`nInstalling requirements..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo activate the environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nTo deactivate, run:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host "`nFor GPU acceleration (NVIDIA):" -ForegroundColor Yellow
Write-Host "  conda install -c conda-forge cudatoolkit numba" -ForegroundColor White
Write-Host "`nTo run the simulations:" -ForegroundColor Yellow
Write-Host "  python run.py          # Boids simulation" -ForegroundColor White
Write-Host "  python run_nbody.py    # N-body simulation" -ForegroundColor White
