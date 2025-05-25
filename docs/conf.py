import os
import tomllib


here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.normpath(os.path.join(here, os.pardir))

with open(os.path.join(project_root, "pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)

project   = "TAPAS"
author    = "Philipp Frech"
release = pyproject["project"]["version"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/pytapas/tapas",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
}
