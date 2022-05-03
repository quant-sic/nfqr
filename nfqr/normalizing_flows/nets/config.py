from typing import List

from pydantic import BaseModel


class NetConfig(BaseModel):

    net_type: str
    net_hidden: List[int]
