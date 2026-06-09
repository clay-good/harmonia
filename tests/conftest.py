import pytest

import harmonia


@pytest.fixture(scope="session")
def ds():
    return harmonia.load()
