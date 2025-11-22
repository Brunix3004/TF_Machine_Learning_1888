import warnings
import pytest

# Ignorar warnings durante los tests
@pytest.fixture(autouse=True)
def ignore_warnings():
    warnings.filterwarnings("ignore")
