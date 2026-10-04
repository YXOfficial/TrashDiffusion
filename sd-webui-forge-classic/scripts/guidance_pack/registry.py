from typing import List, Type
from .base import GuidanceProcessor

_processors: List[GuidanceProcessor] = []

def register_processor(processor_cls: Type[GuidanceProcessor]):
    _processors.append(processor_cls())

def get_processors() -> List[GuidanceProcessor]:
    return _processors
