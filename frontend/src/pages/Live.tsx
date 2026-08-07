import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";

export function Live() {
  const [deviceIndex, setDeviceIndex] = useState<number | null>(null);
  const [streamNonce, setStreamNonce] = useState(0);
  const queryClient = useQueryClient();

  const devices = useQuery({ queryKey: ["webcam-devices"], queryFn: api.listWebcamDevices });
  const cameraId = deviceIndex !== null ? `webcam-${deviceIndex}` : null;

  const status = useQuery({
    queryKey: ["live-status", cameraId],
    queryFn: () => api.getLiveStatus(cameraId!),
    enabled: cameraId !== null,
    refetchInterval: 1000,
  });
  const running = status.data?.running ?? false;

  const start = useMutation({
    mutationFn: () => api.startLive(cameraId!, deviceIndex!),
    onSuccess: () => {
      setStreamNonce((n) => n + 1);
      queryClient.invalidateQueries({ queryKey: ["live-status", cameraId] });
    },
  });

  const stop = useMutation({
    mutationFn: () => api.stopLive(cameraId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["live-status", cameraId] });
      // The recording only finishes writing (and shows up in /videos) once the capture
      // thread has fully stopped and transcoded — refresh the Recordings list to pick it up.
      queryClient.invalidateQueries({ queryKey: ["processed-videos"] });
    },
  });

  // devices.data defaults undefined while loading; only auto-select once loaded.
  if (deviceIndex === null && devices.data && devices.data.length > 0) {
    setDeviceIndex(devices.data[0].device_index);
  }

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Live Camera</h1>

      <div className="mb-4 flex items-end gap-3">
        <div>
          <label className="mb-1 block text-sm text-gray-500">Camera</label>
          <select
            className="rounded-md border border-gray-300 px-3 py-2 text-sm disabled:bg-gray-100"
            value={deviceIndex ?? ""}
            disabled={running || devices.isLoading}
            onChange={(e) => setDeviceIndex(Number(e.target.value))}
          >
            {devices.data?.length === 0 && <option>No cameras detected</option>}
            {devices.data?.map((d) => (
              <option key={d.device_index} value={d.device_index}>
                {d.label}
              </option>
            ))}
          </select>
        </div>

        <button
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          disabled={cameraId === null || running || start.isPending}
          onClick={() => start.mutate()}
        >
          {start.isPending ? "Starting…" : "Start capture"}
        </button>

        <button
          className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-800 disabled:opacity-40"
          disabled={!running || stop.isPending}
          onClick={() => stop.mutate()}
        >
          {stop.isPending ? "Stopping…" : "Stop capture"}
        </button>

        {running && (
          <span className="text-sm text-gray-500">
            {status.data?.frame_count} frames · {status.data?.detection_count} detections
          </span>
        )}
      </div>

      {running && (
        <p className="mb-4 text-sm">
          {status.data?.recognized_names.length ? (
            <span className="font-medium text-green-700">
              Recognized: {status.data.recognized_names.join(", ")}
            </span>
          ) : (
            <span className="text-gray-400">No enrolled employee recognized</span>
          )}
        </p>
      )}

      {status.data?.error && (
        <p className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700">{status.data.error}</p>
      )}
      {start.isError && (
        <p className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700">
          {(start.error as Error).message}
        </p>
      )}

      <div className="flex aspect-video max-w-3xl items-center justify-center rounded-lg border border-gray-200 bg-black">
        {running && cameraId ? (
          <img
            key={streamNonce}
            src={`${api.liveStreamUrl(cameraId)}?t=${streamNonce}`}
            alt="Live processed camera feed"
            className="max-h-full max-w-full"
          />
        ) : (
          <span className="text-sm text-gray-400">No active stream</span>
        )}
      </div>
    </div>
  );
}
