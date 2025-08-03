from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    content: str
    type: str

@dataclass
class ChatGeneration:
    content: str
    type: str
    finish_reason: Optional[str] = None

@dataclass
class Error:
    message: str
    type: type[BaseException]