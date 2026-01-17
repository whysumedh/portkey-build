import axios from 'axios'
import type {
  Project,
  LogStats,
  Recommendation,
  EvaluationRun,
  AnalysisSummary,
  PaginatedResponse,
} from '../types'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Projects
export const projectsApi = {
  list: async (page = 1, pageSize = 20, isActive?: boolean) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
    if (isActive !== undefined) params.set('is_active', String(isActive))
    const { data } = await api.get<PaginatedResponse<Project>>(`/projects?${params}`)
    return data
  },
  
  get: async (id: string) => {
    const { data } = await api.get<Project>(`/projects/${id}`)
    return data
  },
  
  create: async (project: Partial<Project>) => {
    const { data } = await api.post<Project>('/projects', project)
    return data
  },
  
  update: async (id: string, project: Partial<Project>) => {
    const { data } = await api.patch<Project>(`/projects/${id}`, project)
    return data
  },
  
  delete: async (id: string) => {
    await api.delete(`/projects/${id}`)
  },
}

// Logs
export const logsApi = {
  getStats: async (projectId: string) => {
    const { data } = await api.get<LogStats>(`/logs/${projectId}/stats`)
    return data
  },
  
  sync: async (projectId: string, options?: { start_date?: string; end_date?: string; limit?: number }) => {
    const { data } = await api.post(`/logs/${projectId}/sync`, options || {})
    return data
  },
  
  getModels: async (projectId: string) => {
    const { data } = await api.get<string[]>(`/logs/${projectId}/models`)
    return data
  },
}

// Analytics
export const analyticsApi = {
  getSummary: async (projectId: string) => {
    const { data } = await api.get<AnalysisSummary>(`/analytics/${projectId}/summary`)
    return data
  },
  
  runAnalysis: async (projectId: string, type: string, params: Record<string, unknown>) => {
    const { data } = await api.post('/analytics', {
      project_id: projectId,
      type,
      params,
    })
    return data
  },
}

// Evaluations
export const evaluationsApi = {
  list: async (projectId: string, page = 1, pageSize = 20) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
    const { data } = await api.get<PaginatedResponse<EvaluationRun>>(`/evaluations/${projectId}?${params}`)
    return data
  },
  
  get: async (projectId: string, evaluationId: string) => {
    const { data } = await api.get<EvaluationRun>(`/evaluations/${projectId}/${evaluationId}`)
    return data
  },
  
  create: async (projectId: string, candidateModels: string[], sampleSize = 100, timeRangeDays = 30) => {
    const { data } = await api.post<EvaluationRun>('/evaluations', {
      project_id: projectId,
      candidate_models: candidateModels,
      sample_size: sampleSize,
      time_range_days: timeRangeDays,
    })
    return data
  },
  
  cancel: async (projectId: string, evaluationId: string) => {
    const { data } = await api.post(`/evaluations/${projectId}/${evaluationId}/cancel`)
    return data
  },
}

// Recommendations
export const recommendationsApi = {
  list: async (projectId: string, page = 1, pageSize = 20, acknowledged?: boolean) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
    if (acknowledged !== undefined) params.set('acknowledged', String(acknowledged))
    const { data } = await api.get<PaginatedResponse<Recommendation>>(`/recommendations/${projectId}?${params}`)
    return data
  },
  
  getLatest: async (projectId: string) => {
    const { data } = await api.get<Recommendation>(`/recommendations/${projectId}/latest`)
    return data
  },
  
  get: async (projectId: string, recommendationId: string) => {
    const { data } = await api.get<Recommendation>(`/recommendations/${projectId}/${recommendationId}`)
    return data
  },
  
  generate: async (projectId: string, evaluationRunId?: string) => {
    const params = evaluationRunId ? `?evaluation_run_id=${evaluationRunId}` : ''
    const { data } = await api.post<Recommendation>(`/recommendations/${projectId}/generate${params}`)
    return data
  },
  
  acknowledge: async (projectId: string, recommendationId: string, acknowledgedBy: string, actionTaken?: string) => {
    const { data } = await api.post<Recommendation>(`/recommendations/${projectId}/${recommendationId}/acknowledge`, {
      acknowledged_by: acknowledgedBy,
      action_taken: actionTaken,
    })
    return data
  },
}

// Scheduler
export const schedulerApi = {
  getStatus: async () => {
    const { data } = await api.get('/scheduler/status')
    return data
  },
  
  triggerEvaluation: async (projectId: string, candidateModels?: string[], sampleSize?: number) => {
    const { data } = await api.post('/scheduler/trigger', {
      project_id: projectId,
      candidate_models: candidateModels,
      sample_size: sampleSize,
    })
    return data
  },
}

export default api
