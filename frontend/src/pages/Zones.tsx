import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function Zones() {
  const { data, isLoading } = useQuery({ queryKey: ["zones"], queryFn: api.listZones });

  if (isLoading) return <p className="p-6">Loading…</p>;

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Zones</h1>
      <ul className="space-y-2">
        {data?.map((zone) => (
          <li key={zone.zone_id} className="rounded-lg border border-gray-200 p-3">
            <div className="font-medium">{zone.name}</div>
            <div className="text-sm text-gray-500">
              {zone.zone_type} · camera: {zone.camera_id}
            </div>
          </li>
        ))}
      </ul>
      {data?.length === 0 && <p className="text-gray-500">No zones configured.</p>}
    </div>
  );
}
