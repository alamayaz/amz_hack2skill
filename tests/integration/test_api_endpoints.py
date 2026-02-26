"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from rezaa.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestUploadEndpoints:
    def test_upload_audio_no_file(self, client):
        response = client.post("/api/v1/upload/audio")
        assert response.status_code == 422  # Missing file

    def test_upload_video_no_file(self, client):
        response = client.post("/api/v1/upload/video")
        assert response.status_code == 422  # Missing file


class TestStatusEndpoint:
    def test_status_not_found(self, client):
        response = client.get("/api/v1/status/nonexistent-job")
        assert response.status_code == 400  # ValidationError


class TestDownloadEndpoint:
    def test_download_not_found(self, client):
        response = client.get("/api/v1/download/nonexistent-job")
        assert response.status_code == 400


class TestProcessEndpoint:
    def test_process_no_job(self, client):
        response = client.post(
            "/api/v1/process",
            json={"job_id": "nonexistent", "preferences": {}},
        )
        assert response.status_code == 400
