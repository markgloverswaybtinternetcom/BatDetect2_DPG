title train_model Console
set "MODEL=%~dp0Models\model_194_refine14_E61.pth.tar"

REM If first argument (%1) exists, override default
if not "%~1"=="" (
    set "MODEL=%~1"
)

:loop
	uv run "%~dp0train_model.py" "%~dp0TrainingData" "%~dp0ValidationData"  "%MODEL%"
goto loop
pause