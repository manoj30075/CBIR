from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from features import Features
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
features = Features()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/cbir")
async def get_cbir_results(request: Request):
    request_body = await request.json()
    url = request_body['url']
    feature_type = request_body['type']
    return features.get_similar_images_web(url, feature_type)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
