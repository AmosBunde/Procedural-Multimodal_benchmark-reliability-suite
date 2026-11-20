"""The documents must agree with the committed results and with the example code."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numbers_in_the_documents_come_from_the_results(capsys):
    spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts" / "check_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status = module.main()
    assert status == 0, capsys.readouterr().out


def test_readme_shows_the_adapter_example_as_it_is_in_the_repository():
    source = (ROOT / "examples" / "my_adapter.py").read_text()
    start = source.index("class HFClip(Adapter):")
    end = source.index('if __name__ == "__main__":')
    assert source[start:end].strip() in (ROOT / "README.md").read_text()
