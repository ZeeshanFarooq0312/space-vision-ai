import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { ZoneDrawer } from "../components/ZoneDrawer";
import type { ProcessedVideo } from "../types";

// Must match video_upload.py's UPLOAD_CAMERA_ID -- one physical camera for now, so a zone drawn
// once against any upload's preview frame applies to every upload processed afterward.
const UPLOAD_CAMERA_ID = "upload-cam-1";

// track_id is "{camera_id}-{n}" (see tracking/local_tracker.TrackTrackLocalTracker) -- n is
// TrackTrack's own auto-incrementing counter for that processing run (1st, 2nd, 3rd... track it
// ever assigned), not a stable per-person id across runs. The camera_id prefix is redundant here
// (every crop on this page is the same camera), so strip it down to just the number for a
// readable label -- falls back to the raw id if it doesn't match the expected shape.
function formatTrackId(trackId: string): string {
  const match = trackId.match(/-(\d+)$/);
  return match ? `Track #${match[1]}` : trackId;
}

export function Recordings() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<ProcessedVideo | null>(null);
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["processed-videos"],
    queryFn: api.listProcessedVideos,
    refetchInterval: 5000,
  });

  const job = useQuery({
    queryKey: ["upload-status", uploadJobId],
    queryFn: () => api.getUploadStatus(uploadJobId as string),
    enabled: uploadJobId !== null,
    refetchInterval: (query) => (query.state.data?.status === "processing" ? 1500 : false),
  });

  const zones = useQuery({ queryKey: ["zones"], queryFn: api.listZones, enabled: uploadJobId !== null });

  // Once the just-uploaded video finishes, pull it into the list and select it -- its
  // zone_results (people found per zone, with crops) show up in the same detail panel used for
  // any other recording, not a separate one-off view.
  useEffect(() => {
    if (job.data?.status !== "done" || uploadJobId === null) return;
    queryClient.invalidateQueries({ queryKey: ["processed-videos"] }).then(() => {
      const video = queryClient
        .getQueryData<ProcessedVideo[]>(["processed-videos"])
        ?.find((v) => v.video_id === uploadJobId);
      if (video) setSelected(video);
    });
  }, [job.data?.status, uploadJobId, queryClient]);

  if (isLoading) return <p className="p-6">Loading…</p>;

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Recordings</h1>

      <UploadControls onJobStarted={setUploadJobId} isBusy={job.data?.status === "processing"} />

      {uploadJobId && (
        <div className="mb-6">
          <p className="mb-2 text-sm text-gray-500">
            {job.data?.status === "processing"
              ? `Processing… ${job.data.frame_count} frames, ${job.data.detection_count} detections`
              : job.data?.status === "error"
                ? `Failed: ${job.data.error}`
                : "Draw a zone against this frame to enable zone-based login -- since uploads " +
                  "share one camera for now, it applies to every future upload too, not just this one."}
          </p>
          <ZoneDrawer
            cameraId={UPLOAD_CAMERA_ID}
            streamSrc={
              job.data?.status === "processing"
                ? api.uploadStreamUrl(uploadJobId)
                : api.uploadPreviewFrameUrl(uploadJobId)
            }
            frameWidth={job.data?.frame_width ?? null}
            frameHeight={job.data?.frame_height ?? null}
            existingZones={zones.data ?? []}
            large
          />
        </div>
      )}

      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6">
        <div>
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
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-medium text-gray-700">{selected.filename}</div>
                <a
                  href={api.processedVideoUrl(selected.video_id)}
                  download={selected.filename}
                  className="flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    className="h-4 w-4"
                  >
                    <path d="M10 12.5a.75.75 0 0 0 .75-.75V4a.75.75 0 0 0-1.5 0v7.75c0 .414.336.75.75.75Z" />
                    <path d="M5.22 9.72a.75.75 0 0 1 1.06 0L10 13.44l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 10.78a.75.75 0 0 1 0-1.06Z" />
                    <path d="M3.5 15.25a.75.75 0 0 1 .75.75v.5c0 .414.336.75.75.75h10a.75.75 0 0 0 .75-.75v-.5a.75.75 0 0 1 1.5 0v.5A2.25 2.25 0 0 1 15 18.75H5A2.25 2.25 0 0 1 2.75 16.5v-.5a.75.75 0 0 1 .75-.75Z" />
                  </svg>
                  Download
                </a>
              </div>
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
                    download={selected.filename}
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
              {selected.zone_results.length > 0 && (
                <div className="mt-4">
                  <div className="mb-2 text-sm text-gray-500">Zone results</div>
                  <div className="space-y-3">
                    {selected.zone_results.map((zr) => (
                      <div key={zr.zone_id} className="rounded-lg border border-gray-200 p-3">
                        <div className="text-sm font-medium">
                          {zr.zone_name} · {zr.person_count} {zr.person_count === 1 ? "person" : "people"}
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {zr.visits.map((visit, i) => (
                            <div key={i} className="w-20 text-center">
                              {visit.crop_url ? (
                                <img
                                  src={visit.crop_url}
                                  alt={visit.employee_name ?? visit.track_id}
                                  className="h-24 w-20 rounded border border-gray-200 object-cover"
                                />
                              ) : (
                                <div className="flex h-24 w-20 items-center justify-center rounded border border-gray-200 bg-gray-50 text-xs text-gray-400">
                                  no crop
                                </div>
                              )}
                              <div className="mt-1 text-xs font-medium text-gray-700" title={visit.track_id}>
                                {visit.employee_name ?? formatTrackId(visit.track_id)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
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
    </div>
  );
}

function UploadControls({
  onJobStarted,
  isBusy,
}: {
  onJobStarted: (videoId: string) => void;
  isBusy: boolean;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [sampleFps, setSampleFps] = useState(4);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadError(null);
    setUploadingName(file.name);
    try {
      const res = await api.uploadVideo(file, sampleFps);
      onJobStarted(res.video_id);
    } catch (err) {
      setUploadingName(null);
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    }
  }

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
        {uploadingName && <span className="text-sm text-gray-500">{uploadingName}</span>}
      </div>
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
