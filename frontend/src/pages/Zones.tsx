import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";

export function Zones() {
  const { data, isLoading } = useQuery({ queryKey: ["zones"], queryFn: api.listZones });
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const bulkDelete = useMutation({
    mutationFn: (ids: string[]) => Promise.all(ids.map((id) => api.deleteZone(id))).then(() => undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["zones"] });
      setSelected(new Set());
    },
  });

  if (isLoading) return <p className="p-6">Loading…</p>;

  const zones = data ?? [];
  const allSelected = zones.length > 0 && selected.size === zones.length;

  function toggle(zoneId: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(zoneId)) next.delete(zoneId);
      else next.add(zoneId);
      return next;
    });
  }

  function toggleAll() {
    setSelected(allSelected ? new Set() : new Set(zones.map((z) => z.zone_id)));
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Zones</h1>
        {zones.length > 0 && (
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-sm text-gray-600">
              <input type="checkbox" checked={allSelected} onChange={toggleAll} />
              Select all
            </label>
            <button
              className="rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100 disabled:opacity-40"
              disabled={selected.size === 0 || bulkDelete.isPending}
              onClick={() => {
                if (confirm(`Delete ${selected.size} selected zone(s)? This cannot be undone.`)) {
                  bulkDelete.mutate([...selected]);
                }
              }}
            >
              {bulkDelete.isPending ? "Deleting…" : `Delete selected (${selected.size})`}
            </button>
            <button
              className="rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-40"
              disabled={bulkDelete.isPending}
              onClick={() => {
                if (confirm(`Delete ALL ${zones.length} zones? This cannot be undone.`)) {
                  bulkDelete.mutate(zones.map((z) => z.zone_id));
                }
              }}
            >
              Delete all
            </button>
          </div>
        )}
      </div>

      <ul className="space-y-2">
        {zones.map((zone) => (
          <li
            key={zone.zone_id}
            className="flex items-center gap-3 rounded-lg border border-gray-200 p-3"
          >
            <input
              type="checkbox"
              checked={selected.has(zone.zone_id)}
              onChange={() => toggle(zone.zone_id)}
            />
            <div className="flex-1">
              <div className="flex items-center gap-2 font-medium">
                {zone.name}
                {zone.triggers_login && (
                  <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                    Login zone
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-500">
                {zone.zone_type} · camera: {zone.camera_id}
              </div>
            </div>
            <button
              className="rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100 disabled:opacity-40"
              disabled={bulkDelete.isPending}
              onClick={() => {
                if (confirm(`Delete "${zone.name}"? This cannot be undone.`)) {
                  bulkDelete.mutate([zone.zone_id]);
                }
              }}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
      {zones.length === 0 && <p className="text-gray-500">No zones configured.</p>}
    </div>
  );
}
