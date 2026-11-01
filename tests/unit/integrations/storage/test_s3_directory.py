"""Directory downloads must stay within one S3 key prefix and skip markers."""

from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from plexe.integrations.storage.s3 import S3Helper


@pytest.fixture
def storage(monkeypatch):
    """Use a real paginator with stubbed responses and local download receipts."""
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: client)
    downloads = []

    def download(bucket, key, filename):
        downloads.append((bucket, key, filename))
        Path(filename).write_text(key, encoding="utf-8")

    monkeypatch.setattr(client, "download_file", download)
    with Stubber(client) as stubber:
        yield S3Helper("configured-bucket"), stubber, downloads


@pytest.mark.parametrize("suffix", ["", "/"])
def test_directory_listing_uses_a_path_boundary(storage, tmp_path, suffix):
    """A dataset prefix must not include sibling keys such as dataset-old."""
    helper, stubber, downloads = storage
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "exports/dataset/part-0.parquet"}]},
        {"Bucket": "source-bucket", "Prefix": "exports/dataset/"},
    )
    target = helper.download_directory(f"s3://source-bucket/exports/dataset{suffix}", tmp_path)
    assert target == str(tmp_path / "dataset")
    assert downloads == [
        ("source-bucket", "exports/dataset/part-0.parquet", str(tmp_path / "dataset" / "part-0.parquet"))
    ]
    assert (Path(target) / "part-0.parquet").read_text() == "exports/dataset/part-0.parquet"


def test_directory_markers_are_not_downloaded_as_files(storage, tmp_path):
    """Nested folder marker objects cannot obstruct their real child files."""
    helper, stubber, downloads = storage
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [
                {"Key": "dataset/"},
                {"Key": "dataset/partition/"},
                {"Key": "dataset/partition/part.parquet"},
            ]
        },
        {"Bucket": "source-bucket", "Prefix": ANY},
    )
    target = Path(helper.download_directory("s3://source-bucket/dataset", tmp_path))
    assert [key for _, key, _ in downloads] == ["dataset/partition/part.parquet"]
    assert (target / "partition" / "part.parquet").is_file()


def test_directory_download_preserves_pagination(storage, tmp_path):
    """Continue beyond S3's 1,000-key page without dropping the last part."""
    helper, stubber, downloads = storage
    keys = [f"dataset/part-{index:04d}.parquet" for index in range(1001)]
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": key} for key in keys[:1000]], "IsTruncated": True, "NextContinuationToken": "page2"},
        {"Bucket": "source-bucket", "Prefix": ANY},
    )
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": keys[-1]}], "IsTruncated": False},
        {"Bucket": "source-bucket", "Prefix": ANY, "ContinuationToken": "page2"},
    )
    target = Path(helper.download_directory("s3://source-bucket/dataset/", tmp_path))
    assert [key for _, key, _ in downloads] == keys
    assert len(list(target.glob("*.parquet"))) == 1001


@pytest.mark.parametrize("contents", [None, [{"Key": "dataset/"}], [{"Key": "dataset/nested/"}]])
def test_directory_without_files_keeps_not_found_behavior(storage, tmp_path, contents):
    """Empty listings and marker-only directories contain no downloadable files."""
    helper, stubber, downloads = storage
    response = {} if contents is None else {"Contents": contents}
    stubber.add_response("list_objects_v2", response, {"Bucket": "source-bucket", "Prefix": ANY})
    with pytest.raises(ValueError, match="No objects found"):
        helper.download_directory("s3://source-bucket/dataset/", tmp_path)
    assert downloads == []


def test_directory_access_errors_are_preserved(storage, tmp_path):
    """A failed listing is an access error, not an empty dataset."""
    helper, stubber, downloads = storage
    stubber.add_client_error(
        "list_objects_v2",
        service_error_code="AccessDenied",
        http_status_code=403,
        expected_params={"Bucket": "source-bucket", "Prefix": ANY},
    )
    with pytest.raises(ClientError, match="AccessDenied"):
        helper.download_directory("s3://source-bucket/dataset/", tmp_path)
    assert downloads == []
