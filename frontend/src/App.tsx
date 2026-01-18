import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import ProjectList from './pages/ProjectList'
import ProjectDetail from './pages/ProjectDetail'
import ProjectCreate from './pages/ProjectCreate'
import Recommendations from './pages/Recommendations'
import Analytics from './pages/Analytics'
import PortkeyLogs from './pages/PortkeyLogs'
import EvaluationResults from './pages/EvaluationResults'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/" element={<Layout />}>
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="logs" element={<PortkeyLogs />} />
          <Route path="projects" element={<ProjectList />} />
          <Route path="projects/new" element={<ProjectCreate />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="projects/:id/recommendations" element={<Recommendations />} />
          <Route path="projects/:id/analytics" element={<Analytics />} />
          <Route path="projects/:projectId/evaluations/:evaluationId" element={<EvaluationResults />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
