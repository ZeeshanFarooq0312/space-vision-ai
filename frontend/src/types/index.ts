export interface Employee {
  id: string;
  employee_code: string;
  full_name: string;
  role: string;
  department: string | null;
  active: boolean;
}

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
