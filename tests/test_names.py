import pytest

from regenbench.names import validate_name


@pytest.mark.parametrize("name", ["main", "vision-seed0", "a.b_c-1", "X9"])
def test_valid_names_pass(name):
    assert validate_name(name) == name


@pytest.mark.parametrize("name", ["", "..", ".", "a/b", "../x", "a b", "a\n", "run;rm", "é", None, 3])
def test_unsafe_names_are_rejected(name):
    with pytest.raises(ValueError):
        validate_name(name)
