import pytest

from flywheel.text import normalize_text


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("好久不见!你最近怎么样?", "好久不见！你最近怎么样？"),  # noqa: RUF001
        ("  Hello   there ", "hello there"),
        ("Hello.", "hello"),
        ("Hello!!!", "Hello"),
        ("Done…", "done"),
    ],
)
def test_equivalent_answers_normalize_the_same(left, right):
    assert normalize_text(left) == normalize_text(right)


def test_different_answers_stay_different():
    assert normalize_text("Asa ang merkado?") != normalize_text("Hain ang merkado?")


def test_only_trailing_punctuation_is_removed():
    assert normalize_text("Wait, what? Now.") == "wait, what? now"
