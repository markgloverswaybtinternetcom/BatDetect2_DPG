title validate_model Console
set "MODEL=%~dp0models"

REM If first argument (%1) exists, override default
if not "%~1"=="" (
    set "MODEL=%~1"
)
uv run "%~dp0validate_model.py" "%~dp0ValidationData" "%MODEL%"

pause