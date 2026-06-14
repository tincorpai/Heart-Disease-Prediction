from fastapi import FastAPI


app = FastAPI()


@app.get("/")  #This create a single endpoint that responds to HTTP GET requests at the root path by decorating the home() function
def home():
  return {"message": "Welcome to the Randomizer API"}
