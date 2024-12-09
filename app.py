import time
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from config import SECRET_KEY
from cookies import SecureCookieManager, get_current_user
from fastapi import HTTPException

from loguru import logger
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


logger.add("debug.log", level="DEBUG")
logger.add("error.log", level="ERROR")
logger.add("info.log", level="INFO")

app = FastAPI()

# Add SessionMiddleware with a secret key
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,  # Use a strong secret key in production
)

# Optional: Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    url = str(request.url).replace(str(request.base_url), "")
    user_id = request.cookies.get("user_id")
    if user_id != None:
        user_id = SecureCookieManager().decode_secure_cookie(user_id)
    logger.info("User ID: {} ", user_id)
    logger.info("Request: {} path {} ", request.method, url)
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Processed {url} in {process_time} seconds")
    return response


# Jinja2 templates
templates = Jinja2Templates(directory="templates")

from typing import Optional
from fastapi import HTTPException


# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


from routes.user import router as userRouter
from routes.chat import router as chatRouter


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return RedirectResponse(url="/chat")


app.include_router(userRouter)
app.include_router(chatRouter)


from fastapi.responses import JSONResponse, Response
from typing import Literal
import pandas as pd
from io import StringIO


@app.get("/export/chat")
def export_chat(
    type: Literal["json", "csv"],
    request: Request,
    user_id: int = Depends(get_current_user),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    chat_history = request.session.get("chat_history", [])

    if type == "json":
        return JSONResponse(
            content=chat_history,
            headers={"Content-Disposition": "attachment; filename=chat_export.json"},
        )

    elif type == "csv":

        df = pd.DataFrame(chat_history)
        output = StringIO()
        response = Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=chat_export.csv"},
        )
        output.close()
        return response

    raise HTTPException(status_code=400, detail="Invalid export type")
