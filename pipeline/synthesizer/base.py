
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseSynthesizer(ABC):
    """Abstract base class for audio synthesizers."""
    
    @abstractmethod
    def synthesize(self, *args, **kwargs) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
