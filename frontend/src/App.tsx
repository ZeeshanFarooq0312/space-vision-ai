import { BrowserRouter, Route, Routes } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { Alerts } from "./pages/Alerts";
import { Attendance } from "./pages/Attendance";
import { Dashboard } from "./pages/Dashboard";
import { Zones } from "./pages/Zones";

export function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/attendance" element={<Attendance />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/zones" element={<Zones />} />
      </Routes>
    </BrowserRouter>
  );
}
