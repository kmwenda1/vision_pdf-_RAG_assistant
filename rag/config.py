import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves a secret/config value checking in order:
    1. streamlit.secrets (if Streamlit is running or secrets are loaded)
    2. os.environ / .env
    3. default value
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.environ.get(key, default)
