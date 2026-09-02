"""Unit tests for the cloud storage uploader layer.

Covers the three standalone uploader modules (R2 / S3 / OSS) plus the unified
facade in ``storage_uploader``. All network access is mocked via the per-module
client factories (``get_r2_client`` / ``get_s3_client`` / ``get_oss_client``) so
the suite runs fully offline.

These tests also lock in a real bug fix: ``oss_uploader`` previously caught a
non-existent exception class (``oss_exceptions.OssError``); the correct class is
``oss_exceptions.ServiceError``. The OSS error-path tests fail loudly if that
regression ever returns.
"""

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import alibabacloud_oss_v2.exceptions as oss_exceptions
import pytest
from botocore.exceptions import ClientError

from ptc_agent.utils.storage import oss_uploader, r2_uploader, s3_uploader, storage_uploader


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _make_file(tmp_path: Path, name: str = "sample.bin", size: int = 16) -> Path:
    """Create a small real file so existence/size checks pass."""
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return p


def _client_error(code: str = "AccessDenied") -> ClientError:
    return ClientError({"Error": {"Code": code}}, "PutObject")


# --------------------------------------------------------------------------- #
# R2 uploader (26 tests)
# --------------------------------------------------------------------------- #
class TestR2Uploader:
    @pytest.fixture
    def client(self, monkeypatch):
        c = MagicMock()
        monkeypatch.setattr(r2_uploader, "get_r2_client", lambda: c)
        return c

    # --- upload_file ----------------------------------------------------- #
    def test_upload_file_success(self, client, tmp_path):
        f = _make_file(tmp_path)
        assert r2_uploader.upload_file("images/a.png", str(f)) is True
        client.put_object.assert_called_once()

    def test_upload_file_not_found(self, client, tmp_path):
        assert r2_uploader.upload_file("k", str(tmp_path / "nope.bin")) is False
        client.put_object.assert_not_called()

    def test_upload_file_oversize(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(r2_uploader.R2Config, "MAX_UPLOAD_SIZE", 5)
        f = _make_file(tmp_path, size=10)
        assert r2_uploader.upload_file("k", str(f)) is False
        client.put_object.assert_not_called()

    def test_upload_file_client_error(self, client, tmp_path):
        client.put_object.side_effect = _client_error()
        f = _make_file(tmp_path)
        assert r2_uploader.upload_file("k", str(f)) is False

    def test_upload_file_unexpected_exception(self, client, tmp_path):
        client.put_object.side_effect = RuntimeError("boom")
        f = _make_file(tmp_path)
        assert r2_uploader.upload_file("k", str(f)) is False

    # --- upload_bytes ---------------------------------------------------- #
    def test_upload_bytes_success(self, client):
        assert r2_uploader.upload_bytes("text/a.txt", b"hello") is True
        client.put_object.assert_called_once()

    def test_upload_bytes_oversize(self, client, monkeypatch):
        monkeypatch.setattr(r2_uploader.R2Config, "MAX_UPLOAD_SIZE", 3)
        assert r2_uploader.upload_bytes("k", b"too-long") is False
        client.put_object.assert_not_called()

    def test_upload_bytes_client_error(self, client):
        client.put_object.side_effect = _client_error()
        assert r2_uploader.upload_bytes("k", b"data") is False

    # --- upload_base64 --------------------------------------------------- #
    def test_upload_base64_success(self, client):
        import base64

        b64 = base64.b64encode(b"imgbytes").decode()
        assert r2_uploader.upload_base64("images/b.png", b64) is True

    def test_upload_base64_strips_data_uri_prefix(self, client):
        import base64

        raw = base64.b64encode(b"imgbytes").decode()
        assert r2_uploader.upload_base64("images/b.png", f"data:image/png;base64,{raw}") is True

    def test_upload_base64_invalid(self, client):
        assert r2_uploader.upload_base64("images/b.png", "not!valid@base64") is False

    # --- does_object_exist ---------------------------------------------- #
    def test_does_object_exist_true(self, client):
        assert r2_uploader.does_object_exist("images/a.png") is True

    def test_does_object_exist_404_false(self, client):
        client.head_object.side_effect = _client_error("404")
        assert r2_uploader.does_object_exist("images/a.png") is False

    def test_does_object_exist_other_error_false(self, client):
        client.head_object.side_effect = _client_error("500")
        assert r2_uploader.does_object_exist("images/a.png") is False

    # --- delete_object --------------------------------------------------- #
    def test_delete_object_success(self, client):
        assert r2_uploader.delete_object("images/a.png") is True

    def test_delete_object_error(self, client):
        client.delete_object.side_effect = _client_error()
        assert r2_uploader.delete_object("images/a.png") is False

    # --- get_public_url -------------------------------------------------- #
    def test_get_public_url_default(self, monkeypatch):
        monkeypatch.setattr(r2_uploader.R2Config, "ACCOUNT_ID", "acct123")
        monkeypatch.setattr(r2_uploader.R2Config, "PUBLIC_URL_BASE", None)
        assert r2_uploader.get_public_url("images/a.png") == "https://pub-acct123.r2.dev/images/a.png"

    def test_get_public_url_custom_domain(self, monkeypatch):
        monkeypatch.setattr(r2_uploader.R2Config, "PUBLIC_URL_BASE", "https://cdn.example.com")
        assert r2_uploader.get_public_url("images/a.png") == "https://cdn.example.com/images/a.png"

    # --- get_signed_url -------------------------------------------------- #
    def test_get_signed_url_success(self, client):
        client.generate_presigned_url.return_value = "https://signed.example.com/a"
        assert r2_uploader.get_signed_url("images/a.png", expires_in=7200) == "https://signed.example.com/a"

    def test_get_signed_url_error(self, client):
        client.generate_presigned_url.side_effect = _client_error()
        assert r2_uploader.get_signed_url("images/a.png") is None

    # --- upload_image / upload_chart ------------------------------------ #
    def test_upload_image_auto_name(self, client, tmp_path):
        f = _make_file(tmp_path, name="photo.png")
        url = r2_uploader.upload_image(str(f))
        assert url is not None and url.endswith(".png")
        assert "images/" in url

    def test_upload_image_custom_name(self, client, tmp_path):
        f = _make_file(tmp_path, name="photo.png")
        url = r2_uploader.upload_image(str(f), prefix="avatars/", custom_name="user1.png")
        assert url == "https://pub-None.r2.dev/avatars/user1.png"

    def test_upload_image_failure_returns_none(self, client, tmp_path):
        client.put_object.side_effect = _client_error()
        f = _make_file(tmp_path)
        assert r2_uploader.upload_image(str(f)) is None

    def test_upload_chart_delegates(self, client, tmp_path):
        f = _make_file(tmp_path, name="chart.png")
        url = r2_uploader.upload_chart(str(f))
        assert url is not None and "charts/" in url

    # --- verify_connection ---------------------------------------------- #
    def test_verify_connection_success(self, client):
        assert r2_uploader.verify_connection() is True

    def test_verify_connection_failure(self, client):
        client.list_objects_v2.side_effect = _client_error()
        assert r2_uploader.verify_connection() is False


# --------------------------------------------------------------------------- #
# S3 uploader (21 tests)
# --------------------------------------------------------------------------- #
class TestS3Uploader:
    @pytest.fixture
    def client(self, monkeypatch):
        c = MagicMock()
        monkeypatch.setattr(s3_uploader, "get_s3_client", lambda: c)
        return c

    def test_upload_file_success(self, client, tmp_path):
        f = _make_file(tmp_path)
        assert s3_uploader.upload_file("images/a.png", str(f)) is True
        client.put_object.assert_called_once()

    def test_upload_file_not_found(self, client, tmp_path):
        assert s3_uploader.upload_file("k", str(tmp_path / "missing.bin")) is False

    def test_upload_file_oversize(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(s3_uploader.S3Config, "MAX_UPLOAD_SIZE", 5)
        f = _make_file(tmp_path, size=10)
        assert s3_uploader.upload_file("k", str(f)) is False

    def test_upload_file_client_error(self, client, tmp_path):
        client.put_object.side_effect = _client_error()
        f = _make_file(tmp_path)
        assert s3_uploader.upload_file("k", str(f)) is False

    def test_upload_file_unexpected_exception(self, client, tmp_path):
        client.put_object.side_effect = ValueError("x")
        f = _make_file(tmp_path)
        assert s3_uploader.upload_file("k", str(f)) is False

    def test_upload_bytes_success(self, client):
        assert s3_uploader.upload_bytes("text/a.txt", b"hi") is True

    def test_upload_bytes_oversize(self, client, monkeypatch):
        monkeypatch.setattr(s3_uploader.S3Config, "MAX_UPLOAD_SIZE", 2)
        assert s3_uploader.upload_bytes("k", b"longer") is False

    def test_upload_bytes_client_error(self, client):
        client.put_object.side_effect = _client_error()
        assert s3_uploader.upload_bytes("k", b"d") is False

    def test_upload_base64_success(self, client):
        import base64

        assert s3_uploader.upload_base64("images/b.png", base64.b64encode(b"x").decode()) is True

    def test_upload_base64_strips_prefix(self, client):
        import base64

        raw = base64.b64encode(b"x").decode()
        assert s3_uploader.upload_base64("images/b.png", f"data:image/png;base64,{raw}") is True

    def test_upload_base64_invalid(self, client):
        assert s3_uploader.upload_base64("images/b.png", "###bad###") is False

    def test_does_object_exist_true(self, client):
        assert s3_uploader.does_object_exist("images/a.png") is True

    def test_does_object_exist_404_false(self, client):
        client.head_object.side_effect = _client_error("404")
        assert s3_uploader.does_object_exist("images/a.png") is False

    def test_does_object_exist_other_error_false(self, client):
        client.head_object.side_effect = _client_error("403")
        assert s3_uploader.does_object_exist("images/a.png") is False

    def test_delete_object_success(self, client):
        assert s3_uploader.delete_object("images/a.png") is True

    def test_delete_object_error(self, client):
        client.delete_object.side_effect = _client_error()
        assert s3_uploader.delete_object("images/a.png") is False

    def test_get_public_url_default(self, monkeypatch):
        monkeypatch.setattr(s3_uploader.S3Config, "BUCKET_NAME", "mybucket")
        monkeypatch.setattr(s3_uploader.S3Config, "REGION", "us-west-2")
        monkeypatch.setattr(s3_uploader.S3Config, "PUBLIC_URL_BASE", None)
        assert s3_uploader.get_public_url("images/a.png") == "https://mybucket.s3.us-west-2.amazonaws.com/images/a.png"

    def test_get_public_url_custom_domain(self, monkeypatch):
        monkeypatch.setattr(s3_uploader.S3Config, "PUBLIC_URL_BASE", "https://cf.example.com")
        assert s3_uploader.get_public_url("images/a.png") == "https://cf.example.com/images/a.png"

    def test_get_signed_url_success(self, client):
        client.generate_presigned_url.return_value = "https://s3-signed/url"
        assert s3_uploader.get_signed_url("private/a.pdf") == "https://s3-signed/url"

    def test_get_signed_url_error(self, client):
        client.generate_presigned_url.side_effect = _client_error()
        assert s3_uploader.get_signed_url("private/a.pdf") is None

    def test_upload_image_auto_name(self, client, tmp_path):
        f = _make_file(tmp_path, name="pic.png")
        url = s3_uploader.upload_image(str(f))
        assert url and url.endswith(".png") and "images/" in url

    def test_upload_image_failure_returns_none(self, client, tmp_path):
        client.put_object.side_effect = _client_error()
        f = _make_file(tmp_path)
        assert s3_uploader.upload_image(str(f)) is None


# --------------------------------------------------------------------------- #
# OSS uploader (13 tests) -- includes the regression-locking exception fix
# --------------------------------------------------------------------------- #
class TestOSSUploader:
    @pytest.fixture
    def client(self, monkeypatch):
        c = MagicMock()
        monkeypatch.setattr(oss_uploader, "get_oss_client", lambda: c)
        return c

    @staticmethod
    def _service_error(status: int | None = None) -> Exception:
        return oss_exceptions.ServiceError(
            status_code=status if status is not None else 0,
            code="TestError",
            request_id="req-1",
            message="boom",
            ec="",
            timestamp="",
            request_target="/x",
        )

    def test_upload_file_success(self, client, tmp_path):
        f = _make_file(tmp_path)
        assert oss_uploader.upload_file("images/a.png", str(f)) is True
        client.put_object.assert_called_once()

    def test_upload_file_not_found(self, client, tmp_path):
        assert oss_uploader.upload_file("k", str(tmp_path / "gone.png")) is False

    def test_upload_file_oversize(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(oss_uploader.OSSConfig, "MAX_UPLOAD_SIZE", 4)
        f = _make_file(tmp_path, size=10)
        assert oss_uploader.upload_file("k", str(f)) is False

    def test_upload_file_service_error_caught(self, client, tmp_path):
        # Regression guard: the except clause must catch the REAL SDK exception
        # (ServiceError), not a non-existent OssError. If the bug regresses this
        # raises AttributeError when the except clause is evaluated.
        client.put_object.side_effect = self._service_error()
        f = _make_file(tmp_path)
        assert oss_uploader.upload_file("k", str(f)) is False

    def test_upload_bytes_success(self, client):
        assert oss_uploader.upload_bytes("text/a.txt", b"hi") is True

    def test_upload_bytes_oversize(self, client, monkeypatch):
        monkeypatch.setattr(oss_uploader.OSSConfig, "MAX_UPLOAD_SIZE", 2)
        assert oss_uploader.upload_bytes("k", b"longdata") is False

    def test_upload_base64_success(self, client):
        import base64

        assert oss_uploader.upload_base64("images/b.png", base64.b64encode(b"x").decode()) is True

    def test_does_object_exist_true(self, client):
        assert oss_uploader.does_object_exist("images/a.png") is True

    def test_does_object_exist_404_false(self, client):
        client.head_object.side_effect = self._service_error(status=404)
        assert oss_uploader.does_object_exist("images/a.png") is False

    def test_delete_object_success(self, client):
        assert oss_uploader.delete_object("images/a.png") is True

    def test_get_public_url(self, monkeypatch):
        monkeypatch.setattr(oss_uploader.OSSConfig, "BUCKET_NAME", "bk")
        monkeypatch.setattr(oss_uploader.OSSConfig, "ENDPOINT", "oss-cn.com")
        assert oss_uploader.get_public_url("images/a.png") == "https://bk.oss-cn.com/images/a.png"

    def test_get_signed_url_success(self, client):
        res = MagicMock()
        res.url = "https://oss-signed/url"
        client.presign.return_value = res
        assert oss_uploader.get_signed_url("private/a.pdf") == "https://oss-signed/url"

    def test_upload_image_auto_name(self, client, tmp_path):
        f = _make_file(tmp_path, name="pic.png")
        url = oss_uploader.upload_image(str(f))
        assert url and url.endswith(".png") and "images/" in url


# --------------------------------------------------------------------------- #
# Unified facade (8 tests)
# --------------------------------------------------------------------------- #
class TestStorageFacade:
    def test_is_storage_enabled_returns_bool(self):
        assert isinstance(storage_uploader.is_storage_enabled(), bool)

    def test_get_provider_name_nonempty(self):
        assert storage_uploader.get_provider_name()

    def test_get_provider_id_matches_default_or_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_PROVIDER", "r2")
        importlib.reload(storage_uploader)
        assert storage_uploader.get_provider_id() == "r2"
        monkeypatch.setenv("STORAGE_PROVIDER", "s3")
        importlib.reload(storage_uploader)

    def test_all_exports_present(self):
        expected = {
            "delete_object",
            "does_object_exist",
            "get_provider_id",
            "get_provider_name",
            "get_public_url",
            "get_signed_url",
            "is_storage_enabled",
            "upload_base64",
            "upload_bytes",
            "upload_chart",
            "upload_file",
            "upload_image",
            "verify_connection",
        }
        assert expected.issubset(set(storage_uploader.__all__))

    def test_upload_file_identity_with_active_provider(self):
        # The facade re-exports the exact function objects of the active provider,
        # proving dispatch wiring is correct.
        active = storage_uploader.get_provider_id()
        provider_mod = {
            "s3": s3_uploader,
            "r2": r2_uploader,
            "oss": oss_uploader,
        }.get(active)
        if provider_mod is not None:
            assert storage_uploader.upload_file is provider_mod.upload_file

    def test_get_public_url_identity_with_active_provider(self):
        active = storage_uploader.get_provider_id()
        provider_mod = {
            "s3": s3_uploader,
            "r2": r2_uploader,
            "oss": oss_uploader,
        }.get(active)
        if provider_mod is not None:
            assert storage_uploader.get_public_url is provider_mod.get_public_url

    def test_all_functions_callable(self):
        for name in storage_uploader.__all__:
            assert callable(getattr(storage_uploader, name))

    def test_none_provider_is_noop(self, monkeypatch):
        monkeypatch.setenv("STORAGE_PROVIDER", "none")
        importlib.reload(storage_uploader)
        try:
            assert storage_uploader.is_storage_enabled() is False
            assert storage_uploader.upload_file("k", "p") is False
            assert storage_uploader.get_public_url("k") == ""
        finally:
            monkeypatch.setenv("STORAGE_PROVIDER", "s3")
            importlib.reload(storage_uploader)
