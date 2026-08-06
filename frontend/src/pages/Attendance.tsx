import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function Attendance() {
  const { data, isLoading } = useQuery({
    queryKey: ["attendance"],
    queryFn: api.listAttendanceEvents,
  });

  if (isLoading) return <p className="p-6">Loading…</p>;

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Attendance Events</h1>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-gray-500">
            <th className="py-2">Employee</th>
            <th>Event</th>
            <th>Camera</th>
            <th>Occurred At</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((event) => (
            <tr key={event.id} className="border-b border-gray-100">
              <td className="py-2">{event.employee_id}</td>
              <td>{event.event_type}</td>
              <td>{event.camera_id}</td>
              <td>{new Date(event.occurred_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {data?.length === 0 && <p className="text-gray-500">No attendance events yet.</p>}
    </div>
  );
}
