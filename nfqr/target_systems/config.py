from pydantic import BaseModel
from nfqr.target_systems.rotor import QuantumRotorConfig

class ActionConfig(BaseModel):

    target_system:str
    action_type :str
    specific_action_config:QuantumRotorConfig