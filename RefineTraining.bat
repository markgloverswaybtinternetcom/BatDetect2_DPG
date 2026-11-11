title Refine Training Console
set "MODEL=%~dp0Models\model_263_E702.pth.tar"

REM If first argument (%1) exists, override default
if not "%~1"=="" (
    set "MODEL=%~1"
)

:loop
	uv run "%~dp0train_model.py" "%~dp0TrainingData" "%~dp0ValidationData"  "%MODEL%" --lr 0.00005 --patience 30
goto loop
pause