import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
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

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
