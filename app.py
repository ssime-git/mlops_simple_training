from fastapi import FastAPI
import os
import subprocess
import json
import threading

app = FastAPI()

def run_training_locally():
    try:
        result = subprocess.run(
            ["modal", "run", "modal_train.py::train_model"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)  # Debugging purpose
        result = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Modal function execution failed: {e.stderr}")
    except json.JSONDecodeError:
        print("Failed to parse Modal function output")

@app.post("/trigger-training")
async def trigger_training():
    if os.environ.get("MODAL_ENVIRONMENT") == "true":
        # When running on Modal, import and use the Modal function directly
        from modal_train import train_model
        result = await train_model.remote()
    else:
        # For local development, use threading to run the Modal function
        thread = threading.Thread(target=run_training_locally)
        thread.start()
        return {"message": "Training triggered locally"}

    return {"message": "Training triggered", "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
