import axios from "axios";
import type {
  Alert,
  AttendanceEvent,
  Camera,
  Employee,
  EmployeeCreate,
  FaceEnrollmentResult,
  FaceEnrollmentStatus,
  LiveSessionStatus,
  ProcessedVideo,
  UploadJobStatus,
  VideoUploadResponse,
  WebcamDevice,
  Zone,
  ZoneCreate,
} from "../types";

const client = axios.create({ baseURL: "/api" });

export const api = {
  health: () => client.get<{ status: string }>("/health").then((r) => r.data),
  listEmployees: () => client.get<Employee[]>("/employees").then((r) => r.data),
  createEmployee: (payload: EmployeeCreate) =>
    client.post<Employee>("/employees", payload).then((r) => r.data),
  enrollFace: (employeeId: string, photos: { front: Blob; left: Blob; right: Blob }) => {
    const form = new FormData();
    form.append("front", photos.front, "front.jpg");
    form.append("left", photos.left, "left.jpg");
    form.append("right", photos.right, "right.jpg");
    return client
      .post<FaceEnrollmentResult>(`/employees/${employeeId}/enroll-face`, form)
      .then((r) => r.data);
  },
  deleteEmployee: (employeeId: string) => client.delete(`/employees/${employeeId}`).then(() => undefined),
  getFaceEnrollment: (employeeId: string) =>
    client.get<FaceEnrollmentStatus>(`/employees/${employeeId}/face-enrollment`).then((r) => r.data),
  enrollmentPhotoUrl: (employeeId: string, pose: "front" | "left" | "right") =>
    `/api/employees/${employeeId}/photos/${pose}`,
  listCameras: () => client.get<Camera[]>("/cameras").then((r) => r.data),
  listZones: () => client.get<Zone[]>("/zones").then((r) => r.data),
  createZone: (payload: ZoneCreate) => client.post<Zone>("/zones", payload).then((r) => r.data),
  deleteZone: (zoneId: string) => client.delete(`/zones/${zoneId}`).then(() => undefined),
  listAttendanceEvents: () =>
    client.get<AttendanceEvent[]>("/attendance").then((r) => r.data),
  listAlerts: (status?: string) =>
    client
      .get<Alert[]>("/alerts", { params: status ? { status } : undefined })
      .then((r) => r.data),
  listWebcamDevices: () => client.get<WebcamDevice[]>("/live/devices").then((r) => r.data),
  getLiveStatus: (cameraId: string) =>
    client.get<LiveSessionStatus>(`/live/${cameraId}/status`).then((r) => r.data),
  startLive: (cameraId: string, deviceIndex: number, sampleFps = 8.0) =>
    client
      .post<LiveSessionStatus>(`/live/${cameraId}/start`, {
        device_index: deviceIndex,
        sample_fps: sampleFps,
      })
      .then((r) => r.data),
  stopLive: (cameraId: string) =>
    client.post<LiveSessionStatus>(`/live/${cameraId}/stop`).then((r) => r.data),
  liveStreamUrl: (cameraId: string) => `/api/live/${cameraId}/stream`,
  listProcessedVideos: () => client.get<ProcessedVideo[]>("/videos").then((r) => r.data),
  processedVideoUrl: (videoId: string) => `/api/videos/${videoId}/file`,
  uploadVideo: (file: File, sampleFps = 4.0) => {
    const form = new FormData();
    form.append("file", file);
    form.append("sample_fps", String(sampleFps));
    return client
      .post<VideoUploadResponse>("/videos/upload", form)
      .then((r) => r.data);
  },
  getUploadStatus: (videoId: string) =>
    client.get<UploadJobStatus>(`/videos/upload/${videoId}/status`).then((r) => r.data),
  uploadPreviewFrameUrl: (videoId: string) => `/api/videos/upload/${videoId}/preview-frame`,
  uploadStreamUrl: (videoId: string) => `/api/videos/upload/${videoId}/stream`,
};

export default client;
