from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(slots=True)
class ApiResource:
    id: str
    payload: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.payload[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class NemoReconstructClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_pipelines(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/pipelines")

    def upload_video(
        self,
        path: str | Path,
        name: str,
        description: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ApiResource:
        file_path = Path(path)
        payload = {"name": name, "description": description or ""}
        if params:
            payload.update({k: str(v) for k, v in params.items() if v is not None})
        with file_path.open("rb") as file_handle:
            response = self._request(
                "POST",
                "/api/v1/reconstructions/upload",
                files={"file": (file_path.name, file_handle, "video/quicktime")},
                data=payload,
            )
        return ApiResource(id=response["id"], payload=response)

    def list_reconstructions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/reconstructions")

    def get_reconstruction(self, reconstruction_id: str) -> ApiResource:
        response = self._request("GET", f"/api/v1/reconstructions/{reconstruction_id}")
        return ApiResource(id=response["id"], payload=response)

    def get_status(self, reconstruction_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/reconstructions/{reconstruction_id}/status")

    def get_artifacts(self, reconstruction_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/reconstructions/{reconstruction_id}/artifacts")

    def retry_reconstruction(self, reconstruction_id: str, params: dict[str, Any] | None = None) -> ApiResource:
        body = {"params": params} if params else {}
        response = self._request("POST", f"/api/v1/reconstructions/{reconstruction_id}/retry", json=body)
        return ApiResource(id=response["id"], payload=response)

    def delete_reconstruction(self, reconstruction_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/reconstructions/{reconstruction_id}")

    def download_artifact(self, reconstruction_id: str, artifact: str, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(
            self._url(f"/api/v1/reconstructions/{reconstruction_id}/download/{artifact}"),
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return path

    def wait_for_completion(
        self,
        reconstruction_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> ApiResource:
        start = time.time()
        while True:
            resource = self.get_reconstruction(reconstruction_id)
            if resource.status in {"completed", "failed"}:
                return resource
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for reconstruction {reconstruction_id}")
            time.sleep(poll_interval)
