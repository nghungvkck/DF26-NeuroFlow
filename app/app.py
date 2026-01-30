from fastapi import FastAPI
import time
import os
import random

app = FastAPI()

@app.get("/")
def root():
    # giả lập xử lý nặng ngẫu nhiên
    processing_time = random.uniform(0.05, 0.2)
    time.sleep(processing_time)

    return {
        "server": os.getenv("HOSTNAME"),
        "processing_time": round(processing_time, 3),
        "status": "ok"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
