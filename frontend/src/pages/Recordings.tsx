import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api } from "../api/client";
import type { ProcessedVideo } from "../types";

export function Recordings() {
  const [selected, setSelected] = useState<ProcessedVideo | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["processed-videos"],
    queryFn: api.listProcessedVideos,
    refetchInterval: 5000,
  });

  if (isLoading) return <p className="p-6">Loading…</p>;

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6 p-6">
      <div>
        <h1 className="mb-4 text-xl font-semibold">Recordings</h1>
        <VideoUploadPanel />
        <ul className="space-y-2">
          {data?.map((video) => (
            <li key={video.video_id}>
              <button
                onClick={() => setSelected(video)}
                className={`w-full rounded-lg border p-3 text-left ${
                  selected?.video_id === video.video_id
                    ? "border-blue-400 bg-blue-50"
                    : "border-gray-200 hover:bg-gray-50"
                }`}
              >
                <div className="font-medium">{video.camera_id}</div>
                <div className="text-sm text-gray-500">
                  {new Date(video.started_at).toLocaleString()} · {video.duration_seconds.toFixed(1)}s
                </div>
                <div className="mt-1 text-sm text-gray-600">
                  Peak {video.max_people_in_frame} {video.max_people_in_frame === 1 ? "person" : "people"} ·{" "}
                  {video.detection_count} detections across {video.frame_count} frames
                </div>
                {!video.browser_playable && (
                  <div className="mt-1 text-xs text-amber-600">
                    Not browser-playable — download to view
                  </div>
                )}
              </button>
            </li>
          ))}
        </ul>
        {data?.length === 0 && <p className="text-gray-500">No recordings yet — start a live capture first.</p>}
      </div>

      <div>
        {selected ? (
          <div className="sticky top-6">
            <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-gray-200 bg-black">
              {selected.browser_playable ? (
                <video
                  key={selected.video_id}
                  controls
                  autoPlay
                  className="max-h-full max-w-full"
                  src={api.processedVideoUrl(selected.video_id)}
                />
              ) : (
                <a
                  className="text-sm text-blue-400 underline"
                  href={api.processedVideoUrl(selected.video_id)}
                  download
                >
                  Download {selected.filename}
                </a>
              )}
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <Stat label="Camera" value={selected.camera_id} />
              <Stat label="Duration" value={`${selected.duration_seconds.toFixed(1)}s`} />
              <Stat label="Peak people in frame" value={selected.max_people_in_frame} />
              <Stat label="Total detections" value={selected.detection_count} />
              <Stat label="Frames" value={selected.frame_count} />
              <Stat label="Started" value={new Date(selected.started_at).toLocaleString()} />
            </dl>
            {selected.recognized_names.length > 0 && (
              <div className="mt-3 text-sm">
                <div className="text-gray-500">Recognized</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {selected.recognized_names.map((name) => (
                    <span key={name} className="rounded-full bg-green-50 px-2 py-0.5 text-green-700">
                      {name}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed border-gray-300 text-sm text-gray-400">
            Select a recording to preview
          </div>
        )}
      </div>
    </div>
  );
}

function VideoUploadPanel() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [sampleFps, setSampleFps] = useState(4);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { data: job } = useQuery({
    queryKey: ["upload-status", jobId],
    queryFn: () => api.getUploadStatus(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (query.state.data?.status === "processing" ? 1500 : false),
  });

  if (job && job.status !== "processing" && uploadingName !== null) {
    // Job just finished (done or error) -- refresh the recordings list so a
    // successful upload shows up without waiting for the 5s background poll.
    if (job.status === "done") queryClient.invalidateQueries({ queryKey: ["processed-videos"] });
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadError(null);
    setUploadingName(file.name);
    try {
      const res = await api.uploadVideo(file, sampleFps);
      setJobId(res.video_id);
    } catch (err) {
      setUploadingName(null);
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  const isBusy = job?.status === "processing";

  return (
    <div className="mb-4 rounded-lg border border-gray-200 p-3">
      <div className="mb-2 text-sm font-medium">Upload a CCTV video</div>
      <p className="mb-2 text-xs text-gray-500">
        Runs the same detection + recognition pipeline over a pre-recorded file -- useful for
        diagnosing recognition behavior against higher-quality footage than a live webcam.
      </p>
      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500">
          Sample FPS
          <input
            type="number"
            min={1}
            max={30}
            step={1}
            value={sampleFps}
            disabled={isBusy}
            onChange={(e) => setSampleFps(Number(e.target.value) || 4)}
            className="ml-1 w-14 rounded border border-gray-300 px-1 py-0.5"
          />
        </label>
        <button
          type="button"
          disabled={isBusy}
          onClick={() => fileInputRef.current?.click()}
          className="rounded-md border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50 disabled:opacity-50"
        >
          Choose video…
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {uploadingName && (
        <div className="mt-2 text-sm">
          <div className="font-medium">{uploadingName}</div>
          {isBusy && (
            <div className="text-gray-500">
              Processing… {job?.frame_count ?? 0} frames, {job?.detection_count ?? 0} detections
            </div>
          )}
          {job?.status === "done" && <div className="text-green-600">Done -- see it in the list below.</div>}
          {job?.status === "error" && <div className="text-red-600">Failed: {job.error}</div>}
        </div>
      )}
      {uploadError && <div className="mt-2 text-sm text-red-600">{uploadError}</div>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
