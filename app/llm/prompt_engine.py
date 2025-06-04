"""
Prompt Engine - Phase 3
Integrates with Phase 2's VectorStore outputs and Phase 1's MultiTaskAgent.
"""

import os
from pathlib import Path
from typing import Dict
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

class PromptEngine:
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initializes with Phase 2's data path conventions.
        
        Args:
            templates_dir: Optional override (defaults to Phase 2's data sibling)
        """
        self.templates_dir = templates_dir or str(
            Path(__file__).parent.parent.parent / "prompts"  # Matches Phase 2 structure
        )
        self._verify_templates()
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,  # Consistent with Phase 2's whitespace handling
            lstrip_blocks=True
        )

    def _verify_templates(self) -> None:
        """Validates templates exist, mirroring Phase 2's file checks"""
        required = {"tutor_prompt.txt", "quiz_prompt.txt", "recommend_prompt.txt"}
        available = set(os.listdir(self.templates_dir))
        
        if not required.issubset(available):
            missing = required - available
            raise TemplateNotFound(
                f"Missing templates (Phase 3 requirement): {missing}\n"
                f"Found: {available}"
            )

    def render(self, template_name: str, context: Dict) -> str:
        """
        Renders templates using VectorStore's output format from Phase 2.
        
        Args:
            template_name: Matches Phase 1's MultiTaskAgent calls
            context: Expects chunks in Phase 2's format:
                    [{"text": "...", "metadata": {...}}]
                    
        Returns:
            Prompt ready for GeminiClient (Phase 3)
        """
        try:
            template = self.env.get_template(template_name)
            
            # Phase 2 compatibility: Ensure context matches chunk format
            if "context" in context and isinstance(context["context"], list):
                context["chunks"] = [chunk["text"] for chunk in context["context"]]
                
            return template.render(**context).strip()
            
        except TemplateNotFound as e:
            available = os.listdir(self.templates_dir)
            raise TemplateNotFound(
                f"Template {template_name} not found (Phase 3 requirement). "
                f"Available: {available}"
            ) from e