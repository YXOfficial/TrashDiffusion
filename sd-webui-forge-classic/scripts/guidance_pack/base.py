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

    def infotext_fields(self) -> List[str]:
        """Infotext keys, one per component returned by create_ui(), in order.

        The script entry wires these into Script.infotext_fields /
        paste_field_names so Forge's PNG Info "Send to ..." restores our UI.
        Keys must match what process() writes to p.extra_generation_params.
        """
        return []

    def record_params(self, p, params: dict) -> None:
        """Write settings into the PNG infotext (no-op if unavailable)."""
        extra = getattr(p, "extra_generation_params", None)
        if isinstance(extra, dict):
            extra.update(params)
