from datetime import datetime, timedelta
from fastapi import Cookie
from itsdangerous import URLSafeSerializer
from app import SECRET_KEY


class SecureCookieManager:
    def __init__(self):
        self.serializer = URLSafeSerializer(SECRET_KEY, salt="cookie-salt")

    def create_secure_cookie(
        self, user_id: int, expires_in: timedelta = timedelta(days=7)
    ):
        payload = {
            "user_id": user_id,
            "expires": (datetime.utcnow() + expires_in).timestamp(),
        }
        return self.serializer.dumps(payload)

    def decode_secure_cookie(self, cookie_value: str) -> int | None:
        try:
            payload = self.serializer.loads(cookie_value)
            if datetime.fromtimestamp(payload["expires"]) < datetime.utcnow():
                return None
            return payload["user_id"]
        except:
            return None


async def get_current_user(user_id: str = Cookie(default=None)):

    cookie_manager = SecureCookieManager()
    decoded_user_id = cookie_manager.decode_secure_cookie(user_id)

    if not user_id:
        return None

    return decoded_user_id
