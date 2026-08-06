import axios from "axios";
import type { Alert, AttendanceEvent, Camera, Employee, Zone } from "../types";

const client = axios.create({ baseURL: "/api" });

export const api = {
  health: () => client.get<{ status: string }>("/health").then((r) => r.data),
  listEmployees: () => client.get<Employee[]>("/employees").then((r) => r.data),
  listCameras: () => client.get<Camera[]>("/cameras").then((r) => r.data),
  listZones: () => client.get<Zone[]>("/zones").then((r) => r.data),
  listAttendanceEvents: () =>
    client.get<AttendanceEvent[]>("/attendance").then((r) => r.data),
  listAlerts: (status?: string) =>
    client
      .get<Alert[]>("/alerts", { params: status ? { status } : undefined })
      .then((r) => r.data),
};

export default client;
