import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Zone } from "../types";

interface ZoneDrawerProps {
  cameraId: string;
  streamSrc: string;
  /** Actual pixel dimensions of the stream, once known (see LiveSessionStatus.frame_width/height
   * -- null until the session's first frame has been captured). Falls back to the <img>'s own
   * naturalWidth/naturalHeight (same thing, read client-side) if this hasn't arrived yet. */
  frameWidth: number | null;
  frameHeight: number | null;
  existingZones: Zone[];
  /** Renders the preview larger (used on the Recordings upload preview, which gets its own
   * full-width section rather than sharing space with a form) -- defaults to the more compact
   * sizing used inline on the Live page. */
  large?: boolean;
}

type Point = [number, number];

// Distinct, high-contrast colors so overlapping/adjacent zone borders stay tellable apart at a
// glance. Assigned by each zone's position in a stable (zone_id-sorted) order, not by
// triggers_login -- so the same zone always gets the same color across reloads, and two login
// zones (or two non-login zones) don't end up visually identical to each other.
const ZONE_COLORS = [
  "#dc2626", // red
  "#2563eb", // blue
  "#16a34a", // green
  "#d97706", // amber
  "#9333ea", // purple
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
  "#ea580c", // orange
  "#4f46e5", // indigo
];

function colorForZoneIndex(index: number): string {
  return ZONE_COLORS[index % ZONE_COLORS.length];
}

/** Lets the user draw a zone polygon directly on the live preview and save it -- see
 * api/routers/zones.py's POST /zones and zones/monitor.DbZoneMonitor for how a saved zone with
 * `triggers_login` feeds into api/live_stream.py's zone-based attendance login.
 *
 * Points are collected in the <img>'s own displayed (CSS) pixel space, then scaled to real
 * frame-pixel space at save time -- the <img> is shown scaled down (aspect-video container) from
 * its actual captured resolution, so a raw click coordinate isn't usable as-is.
 */
export function ZoneDrawer({
  cameraId,
  streamSrc,
  frameWidth,
  frameHeight,
  existingZones,
  large = false,
}: ZoneDrawerProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  const [drawing, setDrawing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [triggersLogin, setTriggersLogin] = useState(true);
  const queryClient = useQueryClient();

  const zonesForCamera = existingZones
    .filter((z) => z.camera_id === cameraId)
    .sort((a, b) => a.zone_id.localeCompare(b.zone_id));

  const save = useMutation({
    mutationFn: () =>
      api.createZone({
        camera_id: cameraId,
        name: name.trim() || "Zone",
        zone_type: "allowed",
        triggers_login: triggersLogin,
        polygon: points.map(([x, y]) => toFramePoint(x, y)),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["zones"] });
      resetDrawing();
    },
  });

  const deleteZones = useMutation({
    mutationFn: (zoneIds: string[]) => Promise.all(zoneIds.map((id) => api.deleteZone(id))).then(() => undefined),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["zones"] }),
  });

  function toFramePoint(x: number, y: number): Point {
    const naturalW = frameWidth ?? imgRef.current?.naturalWidth ?? displaySize.width;
    const naturalH = frameHeight ?? imgRef.current?.naturalHeight ?? displaySize.height;
    const scaleX = displaySize.width > 0 ? naturalW / displaySize.width : 1;
    const scaleY = displaySize.height > 0 ? naturalH / displaySize.height : 1;
    return [Math.round(x * scaleX), Math.round(y * scaleY)];
  }

  function resetDrawing() {
    setDrawing(false);
    setPoints([]);
    setShowForm(false);
    setName("");
    setTriggersLogin(true);
  }

  // Keep the overlay canvas pixel-for-pixel matched to the <img>'s displayed size, so drawn
  // points line up with what's on screen regardless of how the aspect-video container scales it.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const sync = () => setDisplaySize({ width: img.clientWidth, height: img.clientHeight });
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(img);
    return () => observer.disconnect();
  }, [streamSrc]);

  // Redraw the overlay whenever points/size change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawPolygon = (poly: Point[], color: string, closed: boolean, label?: string) => {
      if (poly.length === 0) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color + "33";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (const [x, y] of poly.slice(1)) ctx.lineTo(x, y);
      if (closed) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
      for (const [x, y] of poly) {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
      if (label) {
        const [lx, ly] = poly[0];
        ctx.font = "bold 12px sans-serif";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(lx, ly - 18, textWidth + 8, 18);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, lx + 4, ly - 5);
      }
    };

    // Existing saved zones, scaled from frame-pixel space back to display space, for context --
    // each gets its own color (see ZONE_COLORS) so overlapping/adjacent zones stay distinguishable.
    const naturalW = frameWidth ?? imgRef.current?.naturalWidth ?? displaySize.width;
    const naturalH = frameHeight ?? imgRef.current?.naturalHeight ?? displaySize.height;
    if (naturalW > 0 && naturalH > 0) {
      const scaleX = displaySize.width / naturalW;
      const scaleY = displaySize.height / naturalH;
      zonesForCamera.forEach((zone, i) => {
        const scaled = zone.polygon.map(([x, y]) => [x * scaleX, y * scaleY] as Point);
        const label = zone.triggers_login ? `${zone.name} (login)` : zone.name;
        drawPolygon(scaled, colorForZoneIndex(i), true, label);
      });
    }

    if (drawing) drawPolygon(points, "#2563eb", points.length >= 3);
  }, [points, displaySize, drawing, zonesForCamera, frameWidth, frameHeight]);

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawing) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setPoints((prev) => [...prev, [e.clientX - rect.left, e.clientY - rect.top]]);
  }

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        {!drawing && !showForm && (
          <button
            className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white"
            onClick={() => setDrawing(true)}
          >
            Draw zone
          </button>
        )}
        {drawing && (
          <>
            <span className="text-sm text-gray-500">Click to add points ({points.length} so far)</span>
            <button
              className="rounded-md bg-gray-200 px-3 py-1.5 text-sm font-medium text-gray-800 disabled:opacity-40"
              disabled={points.length === 0}
              onClick={() => setPoints((prev) => prev.slice(0, -1))}
            >
              Undo point
            </button>
            <button
              className="rounded-md bg-green-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
              disabled={points.length < 3}
              onClick={() => {
                setDrawing(false);
                setShowForm(true);
              }}
            >
              Finish shape
            </button>
            <button
              className="rounded-md bg-gray-200 px-3 py-1.5 text-sm font-medium text-gray-800"
              onClick={resetDrawing}
            >
              Cancel
            </button>
          </>
        )}
        {showForm && (
          <div className="flex flex-wrap items-center gap-2">
            <input
              className="rounded-md border border-gray-300 px-2 py-1.5 text-sm"
              placeholder="Zone name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
            <label className="flex items-center gap-1.5 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={triggersLogin}
                onChange={(e) => setTriggersLogin(e.target.checked)}
              />
              Login zone (recognized employees entering it are logged in)
            </label>
            <button
              className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
              disabled={!name.trim() || save.isPending}
              onClick={() => save.mutate()}
            >
              {save.isPending ? "Saving…" : "Save zone"}
            </button>
            <button
              className="rounded-md bg-gray-200 px-3 py-1.5 text-sm font-medium text-gray-800"
              onClick={resetDrawing}
            >
              Cancel
            </button>
          </div>
        )}
      </div>
      {save.isError && (
        <p className="mb-2 rounded-md bg-red-50 p-2 text-sm text-red-700">
          {(save.error as Error).message}
        </p>
      )}

      {zonesForCamera.length > 0 && (
        <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          {zonesForCamera.map((zone, i) => (
            <div key={zone.zone_id} className="flex items-center gap-1.5 text-xs text-gray-600">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colorForZoneIndex(i) }}
              />
              {zone.name}
              {zone.triggers_login && <span className="text-gray-400">(login)</span>}
              <button
                title={`Delete "${zone.name}"`}
                disabled={deleteZones.isPending}
                className="text-gray-400 hover:text-red-600 disabled:opacity-40"
                onClick={() => {
                  if (confirm(`Delete zone "${zone.name}"? This cannot be undone.`)) {
                    deleteZones.mutate([zone.zone_id]);
                  }
                }}
              >
                ✕
              </button>
            </div>
          ))}
          <button
            className="text-xs font-medium text-red-600 hover:text-red-700 disabled:opacity-40"
            disabled={deleteZones.isPending}
            onClick={() => {
              if (confirm(`Delete all ${zonesForCamera.length} zone(s) for this camera? This cannot be undone.`)) {
                deleteZones.mutate(zonesForCamera.map((z) => z.zone_id));
              }
            }}
          >
            {deleteZones.isPending ? "Deleting…" : "Delete all"}
          </button>
        </div>
      )}

      <div className="relative inline-block">
        <img
          ref={imgRef}
          src={streamSrc}
          alt="Live processed camera feed"
          className={`block max-w-full ${large ? "max-h-[85vh]" : "max-h-[70vh]"}`}
          onLoad={() => {
            const img = imgRef.current;
            if (img) setDisplaySize({ width: img.clientWidth, height: img.clientHeight });
          }}
        />
        <canvas
          ref={canvasRef}
          width={displaySize.width}
          height={displaySize.height}
          className="absolute left-0 top-0"
          style={{ cursor: drawing ? "crosshair" : "default" }}
          onClick={handleCanvasClick}
        />
      </div>
    </div>
  );
}
