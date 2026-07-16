title train_model Console
uv run train_model.py "%CD%\TrainingData" "%CD%\Models"
uv run validate_model.py "%CD%\ValidationData" "%CD%\models"
pause