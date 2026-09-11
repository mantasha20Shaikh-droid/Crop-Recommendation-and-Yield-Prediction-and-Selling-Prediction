# Crop Recommendation and Yield Prediction and Selling Prediction Streamlit Application

import importlib.util
import types

spec = importlib.util.spec_from_file_location("apps_module", r"d:\BCA(AI)\Project\Apps.py")
apps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apps)


def test_url_auth_token_round_trip():
    class FakeQueryParams(dict):
        pass

    fake_query = FakeQueryParams()
    fake_st = types.SimpleNamespace(
        query_params=fake_query,
        experimental_get_query_params=lambda: fake_query,
        experimental_set_query_params=lambda **kwargs: fake_query.update(kwargs),
    )

    apps.st = fake_st

    apps.set_url_auth_token("demo-token")
    assert apps.get_url_auth_token() == "demo-token"

    apps.set_url_auth_token(None)
    assert apps.get_url_auth_token() is None
