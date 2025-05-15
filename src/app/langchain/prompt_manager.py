# src/app/langchain/prompt_manager.py
"""
Gestor de templates de prompts con Jinja2.
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Apunta al directorio donde guardas tus .txt de prompts
PROMPTS_PATH = Path(__file__).parent.parent / "prompts"

env = Environment(
    loader=FileSystemLoader(searchpath=str(PROMPTS_PATH)),
    keep_trailing_newline=True,
    autoescape=False
)


def load_prompt(template_name: str) -> str:
    """
    Carga y retorna el contenido del template de prompt.
    Debe pasarse el nombre de archivo, p.ej. 'base_prompt.txt'.
    """
    template = env.get_template(template_name)
    return template.render()
