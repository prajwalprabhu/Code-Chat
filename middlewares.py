from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


from cookies import SecureCookieManager
from models import verify_user


class AuthenticatedStaticFiles(StaticFiles):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        request = Request(scope)

        # Check for user_id cookie
        user_id = request.cookies.get("user_id")
        cookie_manager = SecureCookieManager()
        user_id = cookie_manager.decode_secure_cookie(user_id)
        if not user_id:
            return await JSONResponse(
                status_code=401, content={"error": "Authentication required"}
            ).__call__(scope, receive, send)

        # Get the requested path from scope
        path = scope.get("path", "")
        requested_path = path.lstrip("/")
        # Check if the user is trying to access their own directory
        if not requested_path.startswith(f"uploads/{user_id}/"):
            return await JSONResponse(
                status_code=403, content={"error": "Access denied to this directory"}
            ).__call__(scope, receive, send)

        try:
            # Verify if user exists in database
            user = await verify_user(user_id)

        except Exception as e:
            return await JSONResponse(
                status_code=401, content={"error": "Invalid authentication"}
            ).__call__(scope, receive, send)

        return await super().__call__(scope, receive, send)
