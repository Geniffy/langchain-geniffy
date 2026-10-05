"""The default client names this package, so the Requests page in the Geniffy app shows which integration
made each call; on a geniffy before 0.2.0, which has no name to give, it is a plain client."""
from langchain_geniffy import __version__
from langchain_geniffy._core import made


class _New:
    def __init__(self, integration=None):
        self.integration = integration


class _Old:
    def __init__(self):
        self.integration = None


def test_the_default_client_names_this_package():
    assert made(_New).integration == f"langchain-geniffy/{__version__}"
    assert made(_Old).integration is None
