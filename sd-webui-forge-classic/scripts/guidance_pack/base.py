from abc import ABC, abstractmethod
from typing import List, Any
import gradio as gr

class GuidanceProcessor(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the guidance method."""
        pass

    @abstractmethod
    def create_ui(self) -> List[gr.components.Component]:
        """Create and return the list of Gradio UI components."""
        pass

    @abstractmethod
    def process(self, p, *args) -> None:
        """
        Apply the guidance logic.
        args matches the list of components returned by create_ui.
        """
        pass

    @abstractmethod
    def register_xyz(self, xyz_grid, set_guidance_value_func) -> None:
        """
        Register options for the XYZ Grid script.
        """
        pass
