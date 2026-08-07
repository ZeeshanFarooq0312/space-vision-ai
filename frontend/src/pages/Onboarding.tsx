import { useMutation, useQueryClient } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { ImageCropper } from "../components/ImageCropper";
import type { EnrollmentPose } from "../types";

/** FastAPI error responses carry the real message in `detail`; axios's own `.message` is just
 * "Request failed with status code 422" unless we pull it out ourselves. */
function errorMessage(e: unknown): string {
  if (isAxiosError(e)) {
    const detail = e.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return e instanceof Error ? e.message : "Something went wrong";
}

const POSES: { pose: EnrollmentPose; instruction: string }[] = [
  { pose: "straight", instruction: "Look straight at the camera" },
  { pose: "left", instruction: "Turn your head to the left" },
  { pose: "right", instruction: "Turn your head to the right" },
];

function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")!.drawImage(video, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("capture failed"))), "image/jpeg", 0.9);
  });
}

export function Onboarding() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const queryClient = useQueryClient();

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [photos, setPhotos] = useState<Partial<Record<EnrollmentPose, { blob: Blob; url: string }>>>({});
  // A raw (uncropped) snapshot awaiting the crop step -- captureFrame() fills this in, and it's
  // cleared once the admin confirms or cancels the crop. The face API rejects wide-angle webcam
  // shots as "too far" (see _check_quality upstream), so cropping in tight on the face before
  // upload is mandatory, not optional polish.
  const [cropping, setCropping] = useState<{ pose: EnrollmentPose; url: string } | null>(null);
  const [employeeCode, setEmployeeCode] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState("");
  const [department, setDepartment] = useState("");
  const [done, setDone] = useState(false);

  // Revoke every captured/in-progress-crop object URL on unmount. Retakes and crop
  // confirm/cancel revoke their own previous URL inline — this effect only needs to run once.
  const photosRef = useRef(photos);
  photosRef.current = photos;
  const croppingRef = useRef(cropping);
  croppingRef.current = cropping;
  useEffect(() => {
    return () => {
      Object.values(photosRef.current).forEach((p) => p && URL.revokeObjectURL(p.url));
      if (croppingRef.current) URL.revokeObjectURL(croppingRef.current.url);
    };
  }, []);

  async function startCamera() {
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("This browser doesn't support camera access (getUserMedia unavailable).");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      // videoRef.current is guaranteed non-null here -- the <video> element is always
      // rendered (see JSX below), it's just visually covered by the "Start camera" button
      // until cameraOn flips. Setting srcObject conditionally on a ref that only exists
      // *after* this same state flip was the original bug: the element wasn't mounted yet.
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setCameraOn(true);
    } catch (e) {
      setCameraError(e instanceof Error ? e.message : "Could not access camera");
    }
  }

  async function capture(pose: EnrollmentPose) {
    if (!videoRef.current) return;
    const blob = await captureFrame(videoRef.current);
    setCropping({ pose, url: URL.createObjectURL(blob) });
  }

  function confirmCrop(blob: Blob) {
    if (!cropping) return;
    const { pose, url: rawUrl } = cropping;
    URL.revokeObjectURL(rawUrl);
    const url = URL.createObjectURL(blob);
    setPhotos((prev) => {
      const previous = prev[pose];
      if (previous) URL.revokeObjectURL(previous.url);
      return { ...prev, [pose]: { blob, url } };
    });
    setCropping(null);
  }

  function cancelCrop() {
    if (!cropping) return;
    URL.revokeObjectURL(cropping.url);
    setCropping(null);
  }

  const allPhotosCaptured = POSES.every(({ pose }) => photos[pose] != null);
  const formValid = employeeCode.trim() && fullName.trim() && role.trim();

  const submit = useMutation({
    mutationFn: async () => {
      const employee = await api.createEmployee({
        employee_code: employeeCode.trim(),
        full_name: fullName.trim(),
        role: role.trim(),
        department: department.trim() || undefined,
      });
      // Our "straight" pose maps to the face API's "front" pose; it's otherwise a 1:1 match.
      return api.enrollFace(employee.id, {
        front: photos.straight!.blob,
        left: photos.left!.blob,
        right: photos.right!.blob,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      streamRef.current?.getTracks().forEach((t) => t.stop());
      setCameraOn(false);
      setDone(true);
    },
  });

  if (done) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-green-200 bg-green-50 p-4 text-green-800">
          {fullName} onboarded successfully. Face embedding stored
          {submit.data ? ` (face_id: ${submit.data.face_id})` : ""}.
        </div>
        <button
          className="mt-4 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white"
          onClick={() => {
            setDone(false);
            Object.values(photos).forEach((p) => p && URL.revokeObjectURL(p.url));
            setPhotos({});
            setEmployeeCode("");
            setFullName("");
            setRole("");
            setDepartment("");
          }}
        >
          Onboard another employee
        </button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-6 p-6">
      <div>
        <h1 className="mb-4 text-xl font-semibold">Onboard Employee</h1>
        <div className="space-y-3">
          <Field label="Employee code" value={employeeCode} onChange={setEmployeeCode} />
          <Field label="Full name" value={fullName} onChange={setFullName} />
          <Field label="Role" value={role} onChange={setRole} />
          <Field label="Department (optional)" value={department} onChange={setDepartment} />
        </div>

        <h2 className="mb-2 mt-6 text-sm font-medium text-gray-700">Enrollment photos</h2>
        <div className="flex gap-3">
          {POSES.map(({ pose }) => (
            <div key={pose} className="flex-1">
              <div className="flex aspect-square items-center justify-center overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
                {photos[pose] ? (
                  <img src={photos[pose]!.url} alt={`${pose} pose`} className="h-full w-full object-cover" />
                ) : (
                  <span className="px-2 text-center text-xs text-gray-400">Not captured</span>
                )}
              </div>
              <button
                className="mt-1 w-full rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-40"
                disabled={!cameraOn}
                onClick={() => capture(pose)}
              >
                {photos[pose] ? "Retake" : "Capture"} {pose}
              </button>
            </div>
          ))}
        </div>

        {submit.isError && (
          <p className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700">
            {errorMessage(submit.error)}
          </p>
        )}

        <button
          className="mt-6 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          disabled={!formValid || !allPhotosCaptured || submit.isPending}
          onClick={() => submit.mutate()}
        >
          {submit.isPending ? "Saving…" : "Save employee"}
        </button>
      </div>

      <div>
        <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-gray-200 bg-black">
          <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-contain" />
          {!cameraOn && (
            <button
              className="absolute rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white"
              onClick={startCamera}
            >
              Start camera
            </button>
          )}
        </div>
        {cameraError && <p className="mt-2 text-sm text-red-700">{cameraError}</p>}
        {cameraOn && (
          <p className="mt-2 text-sm text-gray-500">
            {POSES.find(({ pose }) => !photos[pose])?.instruction ?? "All poses captured."}
          </p>
        )}
      </div>

      {cropping && <ImageCropper src={cropping.url} onConfirm={confirmCrop} onCancel={cancelCrop} />}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm text-gray-500">{label}</span>
      <input
        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
