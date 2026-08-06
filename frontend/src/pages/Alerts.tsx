import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

const severityColor: Record<string, string> = {
  low: "text-gray-600",
  medium: "text-amber-600",
  high: "text-red-600",
};

export function Alerts() {
  const { data, isLoading } = useQuery({ queryKey: ["alerts"], queryFn: () => api.listAlerts() });

  if (isLoading) return <p className="p-6">Loading…</p>;

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Alerts</h1>
      <ul className="space-y-2">
        {data?.map((alert) => (
          <li key={alert.id} className="rounded-lg border border-gray-200 p-3">
            <div className={`text-sm font-medium ${severityColor[alert.severity]}`}>
              {alert.alert_type} · {alert.severity}
            </div>
            <div className="text-sm text-gray-700">{alert.message}</div>
            <div className="text-xs text-gray-400">
              {alert.status} · {new Date(alert.created_at).toLocaleString()}
            </div>
          </li>
        ))}
      </ul>
      {data?.length === 0 && <p className="text-gray-500">No alerts.</p>}
    </div>
  );
}
