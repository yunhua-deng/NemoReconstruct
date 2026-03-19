import type {
  PipelineInfo,
  ReconstructionParams,
  ReconstructionArtifacts,
  ReconstructionDetail,
  ReconstructionStatus,
} from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010";
const API_PREFIX = "/api/v1";

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
  return response.json() as Promise<T>;
}

export async function getPipelines(): Promise<PipelineInfo[]> {
  return fetchJson<PipelineInfo[]>(`${API_PREFIX}/pipelines`);
}

export async function getReconstructions(): Promise<ReconstructionDetail[]> {
  return fetchJson<ReconstructionDetail[]>(`${API_PREFIX}/reconstructions`);
}

export async function getReconstruction(id: string): Promise<ReconstructionDetail> {
  return fetchJson<ReconstructionDetail>(`${API_PREFIX}/reconstructions/${id}`);
}

export async function getReconstructionStatus(id: string): Promise<ReconstructionStatus> {
  return fetchJson<ReconstructionStatus>(`${API_PREFIX}/reconstructions/${id}/status`);
}

export async function getArtifacts(id: string): Promise<ReconstructionArtifacts> {
  return fetchJson<ReconstructionArtifacts>(`${API_PREFIX}/reconstructions/${id}/artifacts`);
}

export async function retryReconstruction(id: string, params?: ReconstructionParams): Promise<ReconstructionDetail> {
  return fetchJson<ReconstructionDetail>(`${API_PREFIX}/reconstructions/${id}/retry`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ? { params } : {}),
  });
}

export async function deleteReconstruction(id: string): Promise<void> {
  const response = await fetch(apiUrl(`${API_PREFIX}/reconstructions/${id}`), { method: "DELETE" });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
}

export function uploadVideo(
  file: File,
  name: string,
  description?: string,
  onProgress?: (pct: number) => void,
  params?: ReconstructionParams,
): Promise<ReconstructionDetail> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(`${API_PREFIX}/reconstructions/upload`));

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as ReconstructionDetail);
      } else {
        reject(new Error(`Upload failed (${xhr.status}): ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed due to a network error"));

    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    if (description) {
      form.append("description", description);
    }
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          form.append(key, String(value));
        }
      }
    }
    xhr.send(form);
  });
}

export function absoluteArtifactUrl(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }
  return apiUrl(path);
}
