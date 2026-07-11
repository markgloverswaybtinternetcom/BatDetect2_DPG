title train_model Console
uv run train_model.py "%CD%\TrainingData" "%CD%\Models\model1"
uv run validate_model.py "%CD%\ValidationData" "%CD%\Models\model1"
pause