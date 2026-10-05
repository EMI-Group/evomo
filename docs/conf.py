"""Build the documentation without importing PyTorch or optional simulators."""

from pathlib import Path

from sphinx import addnodes
from sphinx.environment.adapters.toctree import TocTree
from sphinx.locale import _

ROOT = Path(__file__).resolve().parents[1]
project = "EvoMO"
author = "EvoMO contributors"
copyright = "2026, EvoMO contributors"
# Read the package version without importing it; Python 3.10 has no tomllib.
release = next(
    line.split('"')[1]
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    if line.startswith("version = ")
)
version = release
language = "en"

extensions = ["myst_parser", "autoapi.extension", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "README.md", "papers", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

autoapi_dirs = [str(ROOT / "src" / "evomo")]
autoapi_python_use_implicit_namespaces = True
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary", "imported-members"]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_add_toctree_entry = False
autoapi_keep_files = False

html_theme = "shibuya"
html_title = f"EvoMO {release} documentation"
html_static_path = ["_static", "images"]
html_css_files = ["languages.css", "branding.css", "tables.css"]
html_theme_options = {
    "github_url": "https://github.com/EMI-Group/evomo",
    "light_logo": "_static/evox_brand_dark.svg",
    "dark_logo": "_static/evox_brand_light.svg",
}


def skip_redundant_members(app, what, name, obj, skip, options):
    """Hide empty algorithm attributes and aliases that collide with modules."""
    # Constructor fields already explain these names. Empty instance-attribute
    # entries repeat them and expose internal state without a useful contract.
    # Keep attributes with their own documentation visible.
    if what == "attribute" and name.startswith("evomo.algorithms.") and not obj.docstring.strip():
        return True
    if what == "function" and name in {"evomo.metrics.gd", "evomo.metrics.hv", "evomo.metrics.igd"}:
        return True
    # The optional-import fallback assigns None to the package's module name.
    if what == "data" and name == "evomo.problems.neuroevolution":
        return True
    return None


def add_language_context(app, pagename, templatename, context, doctree):
    """Render navigation from the selected language's home and pair chapters."""
    chinese = pagename.startswith("zh_CN/")
    english_page = pagename.removeprefix("zh_CN/")
    chinese_page = f"zh_CN/{english_page}"
    paired = english_page in app.env.found_docs and chinese_page in app.env.found_docs
    context["language"] = "zh-CN" if chinese else "en"
    context["docstitle"] = f"EvoMO {release} 文档" if chinese else html_title
    context["evomo_languages"] = [
        {"name": "English", "lang": "en", "page": english_page if paired else "index", "active": paired and not chinese},
        {"name": "简体中文", "lang": "zh-CN", "page": chinese_page if paired else "zh_CN/index", "active": paired and chinese},
    ]
    context["evomo_paired_translation"] = paired
    context["evomo_shared_api"] = pagename.startswith("autoapi/")
    navigation_root = "zh_CN/index" if chinese else "index"
    context["root_doc"] = navigation_root

    def language_toctree(**options):
        adapter = TocTree(app.env)
        options["maxdepth"] = int(options.get("maxdepth") or 0)
        fragments = []
        for tree in app.env.get_doctree(navigation_root).findall(addnodes.toctree):
            # Keep the hidden language tree for Sphinx's document discovery,
            # but do not append it to the English navigation.
            if not chinese and "zh_CN/index" in tree["includefiles"]:
                continue
            resolved = adapter.resolve(pagename, app.builder, tree, **options)
            if resolved is not None:
                fragments.append(app.builder.render_partial(resolved)["fragment"])
        return "".join(fragments)

    context["toctree"] = language_toctree
    # The shared discovery tree orders both languages together. Do not let
    # previous/next or breadcrumb navigation jump across the language boundary.
    language_links = {
        app.builder.get_relative_uri(pagename, name) for name in app.env.found_docs if name.startswith("zh_CN/") == chinese
    }
    for relation in ("prev", "next"):
        if context.get(relation) and context[relation]["link"] not in language_links:
            context[relation] = None
    home_link = app.builder.get_relative_uri(pagename, navigation_root)
    context["parents"] = [
        parent for parent in context.get("parents", []) if parent["link"] in language_links and parent["link"] != home_link
    ]
    if chinese:
        translations = {"On this page": "本页目录", "Previous": "上一页", "Next": "下一页", "Home": "首页"}
        context["_"] = lambda message: translations.get(message, _(message))


def setup(app):
    app.connect("autoapi-skip-member", skip_redundant_members)
    app.connect("html-page-context", add_language_context)
