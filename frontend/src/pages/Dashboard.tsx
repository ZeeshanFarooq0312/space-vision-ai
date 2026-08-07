import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";

export function Dashboard() {
  const employees = useQuery({ queryKey: ["employees"], queryFn: api.listEmployees });
  const cameras = useQuery({ queryKey: ["cameras"], queryFn: api.listCameras });
  const alerts = useQuery({ queryKey: ["alerts", "open"], queryFn: () => api.listAlerts("open") });

  return (
    <div className="p-6">
      <div className="mb-4 grid grid-cols-3 gap-4">
        <StatCard label="Employees" value={employees.data?.length} loading={employees.isLoading} />
        <StatCard label="Cameras" value={cameras.data?.length} loading={cameras.isLoading} />
        <StatCard label="Open Alerts" value={alerts.data?.length} loading={alerts.isLoading} />
      </div>

      <Link
        to="/onboarding"
        className="inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white"
      >
        Onboard Employee
      </Link>
    </div>
  );
}

function StatCard({
  label,
  value,
  loading,
}: {
  label: string;
  value: number | undefined;
  loading: boolean;
}) {
  return (
    <div className="rounded-lg border border-gray-200 p-4">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-semibold">{loading ? "…" : value ?? "—"}</div>
    </div>
  );
}
