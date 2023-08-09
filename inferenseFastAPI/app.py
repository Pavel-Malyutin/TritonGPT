import uvicorn
from fastapi import FastAPI

from exceptions import exception_handler
from router import router


app = FastAPI(title="Model inference",
              description="Generate text",
              version="0.0.1",
              contact={
                  "name": "Pavel",
                  "email": "test@example.com"}
              )

app.exception_handler(exception_handler)
app.include_router(router)


if __name__ == '__main__':
    uvicorn.run("app:app", host="localhost", port=8012, reload=True, log_level="debug")
