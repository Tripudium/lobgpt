from dotenv import load_dotenv

from lobgpt.utils import polars_extensions

# Load environment variables from .env file automatically
load_dotenv()

__all__ = ["polars_extensions"]