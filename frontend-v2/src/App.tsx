import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import SitesView from './views/SitesView'
import LieuView from './views/LieuView'
import TrackerView from './views/TrackerView'
import AnalyticsView from './views/AnalyticsView'
import LogsView from './views/LogsView'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<SitesView />} />
        <Route path="lieu/:lieuId" element={<LieuView />} />
        <Route path="tracker" element={<TrackerView />} />
        <Route path="tracker/:siteId" element={<TrackerView />} />
        <Route path="analytics" element={<AnalyticsView />} />
        <Route path="logs" element={<LogsView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
