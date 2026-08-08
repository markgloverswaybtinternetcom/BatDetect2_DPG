title train_model Console
:loop
	uv run train_model.py "%CD%\TrainingData" "%CD%\Models" "%~dp0\ValidationData"
goto loop
pause