import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import type { Employee } from "../types";

const PHOTO_POSES = ["front", "left", "right"] as const;

export function Employees() {
  const { data, isLoading } = useQuery({ queryKey: ["employees"], queryFn: api.listEmployees });
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const bulkDelete = useMutation({
    mutationFn: (ids: string[]) => Promise.all(ids.map((id) => api.deleteEmployee(id))).then(() => undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      setSelected(new Set());
    },
  });

  if (isLoading) return <p className="p-6">Loading…</p>;

  const employees = data ?? [];
  const allSelected = employees.length > 0 && selected.size === employees.length;

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll() {
    setSelected(allSelected ? new Set() : new Set(employees.map((e) => e.id)));
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Employees</h1>
        {employees.length > 0 && (
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-sm text-gray-600">
              <input type="checkbox" checked={allSelected} onChange={toggleAll} />
              Select all
            </label>
            <button
              className="rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100 disabled:opacity-40"
              disabled={selected.size === 0 || bulkDelete.isPending}
              onClick={() => {
                if (confirm(`Delete ${selected.size} selected employee(s)? This cannot be undone.`)) {
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
                if (confirm(`Delete ALL ${employees.length} employees? This cannot be undone.`)) {
                  bulkDelete.mutate(employees.map((e) => e.id));
                }
              }}
            >
              Delete all
            </button>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {employees.map((employee) => (
          <EmployeeRow
            key={employee.id}
            employee={employee}
            selected={selected.has(employee.id)}
            onToggle={() => toggle(employee.id)}
          />
        ))}
      </div>
      {employees.length === 0 && <p className="text-gray-500">No employees onboarded yet.</p>}
    </div>
  );
}

function EmployeeRow({
  employee,
  selected,
  onToggle,
}: {
  employee: Employee;
  selected: boolean;
  onToggle: () => void;
}) {
  const queryClient = useQueryClient();
  const enrollment = useQuery({
    queryKey: ["face-enrollment", employee.id],
    queryFn: () => api.getFaceEnrollment(employee.id),
  });

  const deleteEmployee = useMutation({
    mutationFn: () => api.deleteEmployee(employee.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["employees"] }),
  });

  return (
    <div className="flex items-center gap-4 rounded-lg border border-gray-200 p-3">
      <input type="checkbox" checked={selected} onChange={onToggle} />

      <div className="flex gap-1">
        {PHOTO_POSES.map((pose) => (
          <img
            key={pose}
            src={api.enrollmentPhotoUrl(employee.id, pose)}
            alt={`${employee.full_name} — ${pose}`}
            className="h-14 w-14 rounded-md border border-gray-200 bg-gray-50 object-cover"
            // No enrollment photo saved yet (or the employee was created but never onboarded a
            // face) -- hide rather than show a broken-image icon.
            onError={(e) => (e.currentTarget.style.visibility = "hidden")}
          />
        ))}
      </div>

      <div className="flex-1">
        <div className="font-medium">{employee.full_name}</div>
        <div className="text-sm text-gray-500">
          {employee.employee_code} · {employee.role}
          {employee.department ? ` · ${employee.department}` : ""}
        </div>
        <div className="mt-1 text-xs">
          {enrollment.isLoading ? (
            <span className="text-gray-400">Checking enrollment…</span>
          ) : enrollment.data?.enrolled ? (
            <span className="text-green-700">Face enrolled — face_id: {enrollment.data.face_id}</span>
          ) : (
            <span className="text-amber-600">Not face-enrolled</span>
          )}
        </div>
      </div>

      <button
        className="rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100 disabled:opacity-40"
        disabled={deleteEmployee.isPending}
        onClick={() => {
          if (confirm(`Delete ${employee.full_name}? This removes their record and enrollment photos.`)) {
            deleteEmployee.mutate();
          }
        }}
      >
        {deleteEmployee.isPending ? "Deleting…" : "Delete"}
      </button>
    </div>
  );
}
