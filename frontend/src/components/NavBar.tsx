import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/attendance", label: "Attendance" },
  { to: "/alerts", label: "Alerts" },
  { to: "/zones", label: "Zones" },
];

export function NavBar() {
  return (
    <nav className="flex gap-4 border-b border-gray-200 px-6 py-4">
      <span className="font-semibold">Vision-Stack AI</span>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.to === "/"}
          className={({ isActive }) =>
            isActive ? "font-medium text-blue-600" : "text-gray-600 hover:text-gray-900"
          }
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
