import pytest

from sentimetro.text import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Muy   bueno\n\n producto ", "Muy bueno producto"),
        ("Bueno<br />Recomendado", "Bueno Recomendado"),
        ("Precio &amp; calidad", "Precio & calidad"),
        ("buenooooooo!!!!!!", "buenooo!!!"),
        ("ver https://example.com/x?y=1 ahora", "ver <url> ahora"),
        ("escribime a juan.perez@mail.com", "escribime a <email>"),
        ("tab\tseparado\u200b", "tab separado"),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_normalizes_unicode_to_nfc() -> None:
    decomposed = "cafe\u0301"  # "e" + combining acute accent
    assert clean_text(decomposed) == "café"
    assert len(clean_text(decomposed)) == 4


def test_clean_text_is_idempotent() -> None:
    text = "  ¡Excelenteeeee!   <b>lo recomiendo</b> http://x.y "
    once = clean_text(text)
    assert clean_text(once) == once
