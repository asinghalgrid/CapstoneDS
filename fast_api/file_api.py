from fastapi import FastAPI, File, UploadFile, Form
from time_series_agent import invoke_agent

# Instantiating the API and the fine-tuned YOLO model
app = FastAPI()

# Creating an API to upload an image and return the annotated image with classes and bounding box predictions
@app.post('/execute_workflow/')
def predict(file: UploadFile=File(...), column: str = Form(...)):
    invoke_agent(file.filename, column)