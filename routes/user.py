import os
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from cookies import SecureCookieManager
from models import User, get_db

templates = Jinja2Templates(directory="templates")

router = APIRouter()


# Signup Page
@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(
        "signup.html", {"request": request, "title": "Signup"}
    )


@router.post("/signup")
def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Username already exists", "title": "Signup"},
        )

    user = User(username=username, hashed_password=password)
    db.add(user)
    db.commit()

    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="user_id", value=str(user.id))
    return response


# Login Page
@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html", {"request": request, "title": "Login"}
    )


@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.username == username, User.hashed_password == password)
        .first()
    )

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials", "title": "Login"},
        )

    cookie_manager = SecureCookieManager()
    secure_cookie = cookie_manager.create_secure_cookie(user.id)

    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(
        key="user_id",
        value=secure_cookie,
        httponly=True,  # Prevents JavaScript access
        # secure=True,    # Only sent over HTTPS
        samesite="lax",  # Provides CSRF protection
        max_age=604800,  # 7 days in seconds
    )
    return response


@router.get("/signout")
def signout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="user_id")
    return response
