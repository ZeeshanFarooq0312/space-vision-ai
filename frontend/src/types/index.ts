export interface Employee {
  id: string;
  employee_code: string;
  full_name: string;
  role: string;
  department: string | null;
  active: boolean;
}

export interface EmployeeCreate {
  employee_code: string;
  full_name: string;
  role: string;
  department?: string;
}

export type EnrollmentPose = "straight" | "left" | "right";

export interface Camera {
  camera_id: string;
  name: string;
  role: "entry" | "zone" | "exit";
  location: string | null;
  active: boolean;
}

export interface Zone {
  zone_id: string;
  camera_id: string;
  name: string;
  zone_type: "allowed" | "restricted" | "exit";
  polygon: [number, number][];
}

export interface AttendanceEvent {
  id: string;
  employee_id: string;
  event_type: "login" | "logout" | "auto_logout";
  camera_id: string;
  confidence: number;
  occurred_at: string;
}

export interface Alert {
  id: string;
  alert_type: string;
  severity: "low" | "medium" | "high";
  message: string;
  status: "open" | "acknowledged" | "resolved";
  created_at: string;
}

export interface WebcamDevice {
  device_index: number;
  label: string;
}

export interface LiveSessionStatus {
  camera_id: string;
  running: boolean;
  frame_count: number;
  detection_count: number;
  error: string | null;
}

export interface ProcessedVideo {
  video_id: string;
  camera_id: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  frame_count: number;
  detection_count: number;
  max_people_in_frame: number;
  filename: string;
  browser_playable: boolean;
}
