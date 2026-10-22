"""
Data utils
"""
from cotiger.core.data.schemas import SeqBatch


def cycle(dataloader):
    """
    cycle dataloader never stop
    """
    while True:
        for data in dataloader:
            yield data