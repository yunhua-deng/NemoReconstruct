"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  absoluteArtifactUrl,
  deleteReconstruction,
  getPipelines,
  getReconstructions,
  retryReconstruction,
  uploadVideo,
} from "@/lib/api";
import type { PipelineInfo, ReconstructionDetail } from "@/lib/types";


function formatTime(value: string | null): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString();
}


function prettyStatus(value: string): string {
  return value.replaceAll("_", " ");
}


export default function ReconstructionDashboard() {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [reconstructions, setReconstructions] = useState<ReconstructionDetail[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [uploadPct, setUploadPct] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const [pipelineData, reconstructionData] = await Promise.all([
      getPipelines(),
      getReconstructions(),
    ]);
    setPipelines(pipelineData);
    setReconstructions(reconstructionData);
  }

  useEffect(() => {
    refresh().catch((err: Error) => setError(err.message));
    const interval = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(interval);
  }, []);

  const activeJobs = useMemo(
    () => reconstructions.filter((item) => !["completed", "failed"].includes(item.status)),
    [reconstructions],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Choose a MOV file before uploading.");
      return;
    }

    setError(null);
    setIsUploading(true);
    setUploadPct(0);
    try {
      await uploadVideo(selectedFile, name || selectedFile.name.replace(/\.[^.]+$/, ""), description || undefined, setUploadPct);
      setSelectedFile(null);
      setName("");
      setDescription("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleRetry(id: string) {
    setError(null);
    try {
      await retryReconstruction(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retry failed");
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await deleteReconstruction(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">NemoReconstruct MVP</p>
          <h1>Upload a video and get a splat reconstruction.</h1>
          <p className="hero-copy">
            This demo is intentionally narrow: one upload form, one reconstruction pipeline,
            and one result artifact (PLY splat) for quick validation.
          </p>
        </div>
        <div className="pipeline-panel">
          <h2>Pipeline</h2>
          {pipelines[0] ? (
            <>
              <p className="pipeline-name">{pipelines[0].name}</p>
              <p>{pipelines[0].description}</p>
              <ul className="step-list">
                {pipelines[0].steps.map((step) => (
                  <li key={step}>{prettyStatus(step)}</li>
                ))}
              </ul>
            </>
          ) : (
            <p>Loading pipeline metadata...</p>
          )}
        </div>
      </section>

      <section className="grid-layout">
        <form className="upload-card" onSubmit={handleSubmit}>
          <h2>New Reconstruction</h2>
          <label>
            <span>Name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="warehouse-walkthrough" />
          </label>
          <label>
            <span>Description</span>
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Short note for agents or operators"
              rows={4}
            />
          </label>
          <label>
            <span>Video</span>
            <input
              type="file"
              accept="video/quicktime,video/mp4,.mov,.mp4,.m4v"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
          </label>
          {selectedFile ? <p className="selected-file">Selected: {selectedFile.name}</p> : null}
          {isUploading ? (
            <div className="progress-block">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
              </div>
              <p>Uploading {uploadPct}%</p>
            </div>
          ) : null}
          <button type="submit" disabled={isUploading}>
            {isUploading ? "Uploading..." : "Start reconstruction"}
          </button>
          {error ? <p className="error-text">{error}</p> : null}
        </form>

        <aside className="status-card">
          <h2>Current Queue</h2>
          <p>{activeJobs.length} job(s) in flight</p>
          <p className="status-note">
            The backend runs a single reconstruction worker to keep GPU workloads serialized for fVDB.
          </p>
          <a href={`${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010"}/docs`} target="_blank" rel="noreferrer">
            Open backend docs
          </a>
        </aside>
      </section>

      <section className="table-card">
        <div className="table-headline">
          <div>
            <h2>Reconstructions</h2>
            <p>Poll every 5 seconds. This is the same data the SDKs return.</p>
          </div>
          <button type="button" onClick={() => refresh().catch((err: Error) => setError(err.message))}>
            Refresh
          </button>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Frames</th>
                <th>Updated</th>
                <th>Artifacts</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {reconstructions.map((item) => (
                <tr key={item.id}>
                  <td>
                    <strong>{item.name}</strong>
                    <div className="subtle-line">{item.source_video_filename}</div>
                  </td>
                  <td>
                    <span className={`status-pill status-${item.status}`}>{prettyStatus(item.status)}</span>
                    {item.error_message ? <div className="error-inline">{item.error_message}</div> : null}
                  </td>
                  <td>
                    <div className="mini-progress">
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${item.processing_pct}%` }} />
                      </div>
                      <span>{item.processing_pct}%</span>
                    </div>
                    <div className="subtle-line">{item.processing_step ?? "waiting"}</div>
                  </td>
                  <td>{item.frame_count ?? "-"}</td>
                  <td>{formatTime(item.updated_at)}</td>
                  <td>
                    <div className="artifact-links">
                      {item.artifact_ply_url ? (
                        <a href={absoluteArtifactUrl(item.artifact_ply_url) ?? undefined} target="_blank" rel="noreferrer">PLY</a>
                      ) : null}
                      {item.artifact_usdz_url ? (
                        <a href={absoluteArtifactUrl(item.artifact_usdz_url) ?? undefined} target="_blank" rel="noreferrer">USDZ</a>
                      ) : null}
                      {item.artifact_bundle_url ? (
                        <a href={absoluteArtifactUrl(item.artifact_bundle_url) ?? undefined} target="_blank" rel="noreferrer">ZIP</a>
                      ) : null}
                      {item.artifact_log_url ? (
                        <a href={absoluteArtifactUrl(item.artifact_log_url) ?? undefined} target="_blank" rel="noreferrer">LOG</a>
                      ) : null}
                    </div>
                  </td>
                  <td>
                    <div className="action-row">
                      <button type="button" onClick={() => handleRetry(item.id)}>
                        Retry
                      </button>
                      <button type="button" className="ghost" onClick={() => handleDelete(item.id)}>
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {!reconstructions.length ? (
                <tr>
                  <td colSpan={7} className="empty-cell">No reconstructions yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
