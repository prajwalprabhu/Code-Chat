# start a subprocess and run python main.py >> watch_log
import subprocess
import uvicorn

import app

with open("watch_log", "a") as log_file:
    subprocess.Popen(
        ["python", "main.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

if __name__ == "__main__":
    uvicorn.run(app.app, host="0.0.0.0", port=8080)
