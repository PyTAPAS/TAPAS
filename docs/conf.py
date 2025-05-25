import os
import tomllib


here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.normpath(os.path.join(here, os.pardir))

with open(os.path.join(project_root, "pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)

project   = "TAPAS"
author    = "Philipp Frech"
release = pyproject["project"]["version"]

html_favicon = "_static/icon_dark.png"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon", 
    "sphinx_tabs.tabs", 
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/pytapas/tapas",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "logo": {
    "image_light": "_static/icon_light.png",
    "image_dark":  "_static/icon_dark.png",
    "text":        f"{project} {release}",
    "alt_text":    "TAPAS Home",
    },
}
