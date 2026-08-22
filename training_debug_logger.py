import os, json, numpy
from datetime import datetime

class DebugLogger:
    def __init__(self, log_dir="debug_logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"debug_{timestamp}.jsonl")

    def log(self, data: dict):
        # Convert tensors → lists
        safe_data = {}
        for k, v in data.items():
            if hasattr(v, "detach"):
                safe_data[k] = v.detach().cpu().numpy().tolist()
            else:
                safe_data[k] = v
        with open(self.log_path, "a") as f:
            f.write(json.dumps(safe_data) + "\n")