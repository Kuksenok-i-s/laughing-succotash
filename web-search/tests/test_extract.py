"""HTML to readable text: the prose, without the navigation or the scripts."""

from __future__ import annotations

from web_search.extract import readable_text, strip_markup, truncate

PAGE = """
<html>
  <head>
    <title>Отчёт ENISA</title>
    <style>body { color: red }</style>
    <script>var tracker = "ignore me";</script>
  </head>
  <body>
    <nav>Главная Контакты</nav>
    <h1>Заголовок</h1>
    <p>Фишинг составил более 40&nbsp;% инцидентов.</p>
    <p>Второй абзац.</p>
    <footer>&copy; 2026</footer>
  </body>
</html>
"""


def test_the_title_and_the_prose_are_separated() -> None:
    title, text = readable_text(PAGE)

    assert title == "Отчёт ENISA"
    assert "Фишинг составил более 40 % инцидентов." in text
    assert "Второй абзац." in text


def test_script_and_style_bodies_never_reach_the_text() -> None:
    """A script body read as prose is not merely noise; it is actively misleading."""
    _, text = readable_text(PAGE)

    assert "tracker" not in text
    assert "ignore me" not in text
    assert "color: red" not in text


def test_navigation_and_footers_never_reach_the_text() -> None:
    """A model asked to check a claim should not have to read past the site menu first."""
    _, text = readable_text(PAGE)

    assert "Главная" not in text
    assert "Контакты" not in text
    assert "2026" not in text


def test_the_marked_main_block_wins_over_the_chrome_around_it() -> None:
    prose = "Настоящий текст статьи. " * 20
    _, text = readable_text(
        f"<body><div>Реклама и хлебные крошки</div>"
        f"<main><p>{prose}</p></main>"
        f"<div>Читайте также</div></body>"
    )

    assert "Настоящий текст статьи." in text
    assert "хлебные крошки" not in text
    assert "Читайте также" not in text


def test_role_main_is_honoured_too() -> None:
    """Wikipedia and the Python docs both mark their content with a role, not a <main>."""
    prose = "Содержимое страницы. " * 20
    _, text = readable_text(
        f"<body><div>меню</div><div role='main'><p>{prose}</p></div></body>"
    )

    assert "Содержимое страницы." in text
    assert "меню" not in text


def test_nesting_inside_the_main_block_does_not_end_it_early() -> None:
    tail = "Последний абзац статьи, который обязан дойти до читателя."
    _, text = readable_text(
        f"<main><div><div><p>начало</p></div></div><p>{tail}</p></main>"
    )

    assert "начало" in text
    assert tail in text


def test_a_tiny_article_teaser_does_not_hijack_the_page() -> None:
    """A sidebar <article> is not the content, so the whole document is the safer answer."""
    prose = "Основной текст без разметки main. " * 20
    _, text = readable_text(
        f"<body><article>анонс</article><div><p>{prose}</p></div></body>"
    )

    assert "Основной текст без разметки main." in text


def test_a_chrome_class_on_a_plain_div_is_dropped() -> None:
    """Wikipedia's language switcher is a <div class="mw-portlet-lang">, not a <nav>."""
    prose = "Текст статьи для проверки. " * 20
    _, text = readable_text(
        f"<body><div class='vector-menu mw-portlet mw-portlet-lang'>"
        f"Afrikaans Aragonés Azərbaycanca</div><div><p>{prose}</p></div></body>"
    )

    assert "Текст статьи для проверки." in text
    assert "Aragonés" not in text


def test_a_chrome_class_on_the_html_element_does_not_discard_the_page() -> None:
    """Wikipedia tags <html> with vector-feature-language-in-main-menu, which contains 'menu'."""
    prose = "Единственный содержательный абзац. " * 20
    _, text = readable_text(
        f"<html class='vector-feature-language-in-main-menu'><body><p>{prose}</p></body></html>"
    )

    assert "Единственный содержательный абзац." in text


def test_a_page_the_heuristic_empties_is_returned_with_its_chrome_instead() -> None:
    """A class-name guess that misfires should cost the tidying, not the page."""
    _, text = readable_text("<div class='sidebar'><p>это всё, что есть на странице</p></div>")

    assert "это всё, что есть на странице" in text


def test_the_title_survives_even_inside_a_chrome_block() -> None:
    title, _ = readable_text(
        "<html><head class='site-header'><title>Имя страницы</title></head></html>"
    )

    assert title == "Имя страницы"


def test_block_tags_separate_paragraphs() -> None:
    _, text = readable_text("<p>первый</p><p>второй</p>")

    assert text == "первый\n\nвторой"


def test_entities_are_decoded() -> None:
    _, text = readable_text("<p>&lt;тег&gt; &amp; &quot;кавычки&quot;</p>")

    assert text == '<тег> & "кавычки"'


def test_runs_of_blank_lines_collapse() -> None:
    _, text = readable_text("<div><div><div>один</div></div></div><p>два</p>")

    assert "\n\n\n" not in text
    assert "один" in text and "два" in text


def test_malformed_markup_returns_what_was_parsed() -> None:
    _, text = readable_text("<p>начало<div><span>и ещё")

    assert "начало" in text
    assert "и ещё" in text


def test_a_document_with_no_body_text_is_empty_not_an_error() -> None:
    title, text = readable_text("<html><head><title>Пусто</title></head><body></body></html>")

    assert title == "Пусто"
    assert text == ""


def test_highlight_markup_is_stripped_from_an_excerpt() -> None:
    assert strip_markup("Phishing <strong>40%</strong> of &gt; incidents") == (
        "Phishing 40% of > incidents"
    )


def test_stripping_an_empty_excerpt_is_safe() -> None:
    assert strip_markup("") == ""
    assert strip_markup(None) == ""  # type: ignore[arg-type]


def test_long_pages_are_truncated_visibly() -> None:
    """Silently dropping the tail would let the agent reason over half a page."""
    result = truncate("а" * 500, 100)

    assert len(result) < 500
    assert "обрезана" in result


def test_short_pages_are_returned_untouched() -> None:
    assert truncate("коротко", 100) == "коротко"
