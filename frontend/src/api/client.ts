import axios from "axios";
import type {
  Alert,
  AttendanceEvent,
  Camera,
  Employee,
  EmployeeCreate,
  EnrollmentPose,
  LiveSessionStatus,
  ProcessedVideo,
  WebcamDevice,
  Zone,
} from "../types";

const client = axios.create({ baseURL: "/api" });

export const api = {
  health: () => client.get<{ status: string }>("/health").then((r) => r.data),
  listEmployees: () => client.get<Employee[]>("/employees").then((r) => r.data),
  createEmployee: (payload: EmployeeCreate) =>
    client.post<Employee>("/employees", payload).then((r) => r.data),
  uploadEnrollmentPhoto: (employeeId: string, pose: EnrollmentPose, blob: Blob) => {
    const form = new FormData();
    form.append("file", blob, `${pose}.jpg`);
    return client.put(`/employees/${employeeId}/photos/${pose}`, form).then(() => undefined);
  },
  listCameras: () => client.get<Camera[]>("/cameras").then((r) => r.data),
  listZones: () => client.get<Zone[]>("/zones").then((r) => r.data),
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
};

export default client;
