title train_model Console
:loop
	uv run train_model.py "%CD%\TrainingData" "%CD%\Models" "%~dp0\ValidationData"
	uv run validate_model.py "%CD%\ValidationData" "%CD%\models"
goto loop
pause