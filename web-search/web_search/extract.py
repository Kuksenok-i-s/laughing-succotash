"""HTML to readable text.

Brave and SearXNG return excerpts already, so this exists only for ``/v1/fetch``: the agent opens
a page when two sources disagree and it needs the actual sentence. A full DOM parser would be a
dependency this unit does not otherwise need, and the goal is not fidelity — it is the prose,
without the navigation, in a bounded number of characters.

Titles and excerpts from the search backends go through ``strip_markup`` instead: Brave marks the
matched terms with ``<strong>``, which is presentation, not content.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

# Content inside these never reads as prose, and script bodies are actively misleading to a model.
#
# The second group is chrome: menus, site footers, sidebars and forms surround the article on every
# page that has one, and a model asked to check a claim should not have to read past them first.
_DROP = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "canvas", "iframe",
        "nav", "footer", "aside", "form", "button", "select", "datalist",
    }
)

# Tags that mark the article itself. When a page says where its content is, the chrome outside it
# can be dropped wholesale rather than guessed at tag by tag.
_MAIN = frozenset({"main", "article"})

# Below this, a marked-up main block is more likely a stray <article> teaser in a sidebar than the
# page's content, and the whole document is the safer answer.
_MAIN_MIN_CHARS = 200

# Chrome that hides behind a <div>. Wikipedia puts its 122-language switcher in
# ``class="mw-portlet-lang"`` and its contents list in ``class="vector-toc-contents"``, neither of
# which any tag name gives away — and on a long article they eat a tenth of the character budget
# before the first sentence. Matching is per class token, not a substring of the whole attribute,
# so ``header`` does not also catch ``subheading``. A false positive costs one block; the article
# itself is not marked with any of these.
_CHROME_TOKENS = frozenset(
    {
        "nav", "navbar", "navigation", "menu", "sidebar", "breadcrumb", "breadcrumbs",
        "toc", "portlet", "banner", "advert", "advertisement", "cookie", "cookies",
        "share", "sharing", "social", "related", "comments", "footer", "siteheader",
        "sitenav", "skip", "pagination", "subscribe", "newsletter", "paywall",
    }
)

# Elements that carry the whole document and so can never be chrome, whatever their class says.
# Wikipedia's <html> is tagged ``vector-feature-language-in-main-menu``, which contains the token
# ``menu`` and would otherwise discard every page it serves.
_STRUCTURAL = frozenset({"html", "body", "main", "article"})

# Elements that never close, and so must not be counted when tracking nesting depth.
_VOID = frozenset(
    {
        "br", "img", "input", "meta", "link", "hr", "source", "track", "wbr",
        "area", "base", "col", "embed", "param",
    }
)

# Tags that end a line of text rather than continuing it.
_BREAK = frozenset(
    {
        "p", "br", "div", "section", "article", "header", "main",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th", "blockquote",
        "pre", "figure", "figcaption", "table", "thead", "tbody", "ul", "ol",
    }
)

_TAG = re.compile(r"<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]{2,}")
# A non-breaking space is a space to every reader of this text, and a distinct codepoint to every
# regex downstream that looks for one.
_NBSP = "\u00a0"


class _Reader(HTMLParser):
    """Accumulates text, dropping the parts of a page that are not prose.

    Text is collected twice: once for the whole document, and once for whatever the page marks as
    its main content. Which of the two is worth returning is decided after parsing, because until
    the end we do not know whether the marked region held anything.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._main_parts: list[str] = []
        # Unfiltered, so that a class-name heuristic which misfires on a whole page costs us the
        # tidying rather than the page.
        self._raw_parts: list[str] = []
        self._suppress = 0
        # Nesting depth inside the main region; 0 means we are outside it.
        self._main_depth = 0
        # Same, for a block whose class or id marks it as chrome.
        self._chrome_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP:
            self._suppress += 1
            return
        if tag == "title":
            self._in_title = True

        if tag not in _VOID:
            if self._chrome_depth:
                self._chrome_depth += 1
            elif tag not in _STRUCTURAL and _is_chrome(attrs):
                self._chrome_depth = 1

        if self._main_depth:
            if tag not in _VOID:
                self._main_depth += 1
        elif tag in _MAIN or ("role", "main") in attrs:
            self._main_depth = 1

        if tag in _BREAK:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP:
            # Clamped at zero: a stray closing tag must not un-suppress a script body.
            self._suppress = max(0, self._suppress - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BREAK:
            self._append("\n")
        if tag not in _VOID:
            if self._main_depth:
                self._main_depth -= 1
            if self._chrome_depth:
                self._chrome_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._suppress:
            return
        if self._in_title:
            # Kept even inside chrome: the <title> is the page's name, wherever it sits.
            self.title += data
            return
        if data.strip():
            self._append(data)

    def _append(self, part: str) -> None:
        self._raw_parts.append(part)
        if self._chrome_depth:
            return
        self._parts.append(part)
        if self._main_depth:
            self._main_parts.append(part)

    def text(self) -> str:
        """The narrowest buffer that still holds a page.

        Tried in order: the marked main block, then the document minus its chrome, then the
        document as it came. Each step is an improvement that might have gone wrong, and a page
        with the menus still in it beats no page at all.
        """
        main = "".join(self._main_parts)
        if len(main.strip()) >= _MAIN_MIN_CHARS:
            return main
        clean = "".join(self._parts)
        if clean.strip():
            return clean
        return "".join(self._raw_parts)


def _is_chrome(attrs: list[tuple[str, str | None]]) -> bool:
    """Whether an element's class, id or role marks it as something other than content."""
    for name, value in attrs:
        if name == "role" and value in ("navigation", "banner", "search", "complementary"):
            return True
        if name in ("class", "id") and value:
            # MediaWiki writes mw-portlet-lang, Vector writes vector-toc-contents: the marker is a
            # part of a hyphenated name rather than a token of its own.
            for token in value.lower().replace("_", "-").split():
                if _CHROME_TOKENS.intersection(token.split("-")):
                    return True
    return False


def readable_text(markup: str) -> tuple[str, str]:
    """``(title, text)`` for one HTML document. Malformed markup yields what was parsed."""
    reader = _Reader()
    try:
        reader.feed(markup)
        reader.close()
    except Exception:
        # html.parser is lenient, but a pathological document should still return partial prose
        # rather than failing the whole fetch.
        pass
    return _collapse(reader.title), _collapse(reader.text())


def strip_markup(value: str) -> str:
    """Plain text from a fragment that may carry highlight tags and entities."""
    return _collapse(html.unescape(_TAG.sub(" ", value or "")))


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[…страница обрезана]"


def _collapse(value: str) -> str:
    lines = [
        _SPACES.sub(" ", line.replace(_NBSP, " ")).strip() for line in value.splitlines()
    ]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
