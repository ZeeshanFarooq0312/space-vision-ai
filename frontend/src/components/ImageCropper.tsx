import { useRef, useState } from "react";

const OUTPUT_SIZE = 480; // px, square output
const MAX_SCALE = 4;

/** Drag-to-pan + zoom cropper over a fixed square viewport. All math is done in the source
 * image's natural pixel space so the final canvas crop exactly matches what's previewed. */
export function ImageCropper({
  src,
  onConfirm,
  onCancel,
}: {
  src: string;
  onConfirm: (blob: Blob) => void;
  onCancel: () => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  // The scale below which the image can't fully cover the OUTPUT_SIZE viewport (leaving a blank
  // gap at an edge) depends on the source resolution, so it's computed on load, not hardcoded.
  const [minScale, setMinScale] = useState(1);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 }); // top-left of the scaled image, in displayed px
  const dragState = useRef<{ startX: number; startY: number; origin: { x: number; y: number } } | null>(null);

  function clamp(next: { x: number; y: number }, s: number, size: { w: number; h: number }) {
    const minX = Math.min(0, OUTPUT_SIZE - size.w * s);
    const minY = Math.min(0, OUTPUT_SIZE - size.h * s);
    return { x: Math.min(0, Math.max(minX, next.x)), y: Math.min(0, Math.max(minY, next.y)) };
  }

  function onImageLoad() {
    const img = imgRef.current!;
    const size = { w: img.naturalWidth, h: img.naturalHeight };
    setNaturalSize(size);
    // start fully zoomed to cover the viewport, centered
    const coverScale = Math.max(OUTPUT_SIZE / size.w, OUTPUT_SIZE / size.h);
    setMinScale(coverScale);
    setScale(coverScale);
    setOffset(
      clamp({ x: (OUTPUT_SIZE - size.w * coverScale) / 2, y: (OUTPUT_SIZE - size.h * coverScale) / 2 }, coverScale, size),
    );
  }

  function onPointerDown(e: React.PointerEvent) {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragState.current = { startX: e.clientX, startY: e.clientY, origin: offset };
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!dragState.current || !naturalSize) return;
    const dx = e.clientX - dragState.current.startX;
    const dy = e.clientY - dragState.current.startY;
    setOffset(clamp({ x: dragState.current.origin.x + dx, y: dragState.current.origin.y + dy }, scale, naturalSize));
  }

  function onPointerUp() {
    dragState.current = null;
  }

  function onScaleChange(next: number) {
    if (!naturalSize) return;
    setScale(next);
    setOffset((prev) => clamp(prev, next, naturalSize));
  }

  function confirm() {
    if (!naturalSize) return;
    const canvas = document.createElement("canvas");
    canvas.width = OUTPUT_SIZE;
    canvas.height = OUTPUT_SIZE;
    const ctx = canvas.getContext("2d")!;
    const sx = -offset.x / scale;
    const sy = -offset.y / scale;
    const sSize = OUTPUT_SIZE / scale;
    ctx.drawImage(imgRef.current!, sx, sy, sSize, sSize, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
    canvas.toBlob((blob) => blob && onConfirm(blob), "image/jpeg", 0.92);
  }

  // hidden until the image has loaded and we know its natural size
  const ready = naturalSize !== null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="rounded-lg bg-white p-4 shadow-xl">
        <p className="mb-2 text-sm text-gray-600">Drag to reposition, zoom to bring the face closer.</p>
        <div
          className="relative touch-none overflow-hidden rounded-md border border-gray-300 bg-gray-100"
          style={{ width: OUTPUT_SIZE, height: OUTPUT_SIZE, cursor: ready ? "grab" : "default" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          {/* eslint-disable-next-line jsx-a11y/alt-text -- internal cropper source, not user-facing content */}
          <img
            ref={imgRef}
            src={src}
            onLoad={onImageLoad}
            draggable={false}
            className="absolute left-0 top-0 select-none"
            style={{
              width: naturalSize?.w,
              height: naturalSize?.h,
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: "top left",
              visibility: ready ? "visible" : "hidden",
            }}
          />
        </div>

        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs text-gray-500">Zoom</span>
          <input
            type="range"
            min={minScale}
            max={Math.max(MAX_SCALE, minScale)}
            step={0.05}
            value={scale}
            disabled={!ready}
            onChange={(e) => onScaleChange(Number(e.target.value))}
            className="flex-1"
          />
        </div>

        <div className="mt-4 flex justify-end gap-2">
          <button
            className="rounded-md bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            disabled={!ready}
            onClick={confirm}
          >
            Use this crop
          </button>
        </div>
      </div>
    </div>
  );
}
