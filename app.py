from signLanguage.logger import logging
from signLanguage.exception import SignException
import sys  # Import sys to pass it to SignException

# Logging a message
logging.info("Welcome to my application")

try:
    a = 7 / "8"  # No quotes around 8; it should be an integer for division
except Exception as e:
    raise SignException(e, sys)
