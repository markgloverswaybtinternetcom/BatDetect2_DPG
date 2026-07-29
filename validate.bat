title validate_model Console
set "MODEL=%~dp0\models"

REM If first argument (%1) exists, override default
if not "%~1"=="" (
    set "MODEL=%~1"
)
uv run "%~dp0\validate_model.py" "%~dp0\ValidationData" "%MODEL%"

pause