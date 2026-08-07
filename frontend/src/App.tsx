import { BrowserRouter, Route, Routes } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { Alerts } from "./pages/Alerts";
import { Attendance } from "./pages/Attendance";
import { Dashboard } from "./pages/Dashboard";
import { Live } from "./pages/Live";
import { Onboarding } from "./pages/Onboarding";
import { Recordings } from "./pages/Recordings";
import { Zones } from "./pages/Zones";

export function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<Live />} />
        <Route path="/recordings" element={<Recordings />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/attendance" element={<Attendance />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/zones" element={<Zones />} />
      </Routes>
    </BrowserRouter>
  );
}
