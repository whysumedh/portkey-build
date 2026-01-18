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

// Portkey Direct Logs
export interface PortkeyLog {
  id: string
  db_id?: string  // Internal database ID
  trace_id?: string
  request_id?: string
  span_id?: string
  created_at?: string
  time_of_generation?: string
  ingested_at?: string
  // Model info - Portkey uses ai_org for provider
  ai_model?: string
  ai_org?: string  // This is the provider
  ai_provider?: string  // Alias
  model?: string  // Alias
  provider?: string  // Alias
  mode?: string
  config?: string
  prompt_slug?: string
  // Tokens - Portkey uses "units" terminology
  req_units?: number  // prompt tokens
  res_units?: number  // completion tokens
  total_units?: number  // total tokens
  prompt_tokens?: number  // Alias
  completion_tokens?: number  // Alias
  total_tokens?: number  // Alias
  // Cost and performance
  cost?: number
  cost_currency?: string
  response_time?: number
  latency_ms?: number  // Alias
  response_status_code?: number
  request_url?: string
  // Status
  status?: string
  is_success?: boolean
  refusal?: boolean
  // Extracted content
  prompt?: string  // Extracted user message
  system_prompt?: string  // Extracted system message
  // Request/Response bodies (full JSON from Portkey)
  request?: Record<string, unknown>
  response?: Record<string, unknown>
  metadata?: Record<string, unknown>
  project_id?: string
}

export interface PortkeyLogsResponse {
  logs: PortkeyLog[]
  total: number
  from_cache: boolean
  last_synced?: string
}

export const portkeyLogsApi = {
  getLogs: async (options?: { 
    workspace_id?: string
    hours?: number
    limit?: number
    refresh?: boolean  // Pass true to fetch new logs from Portkey
    user_filter?: string  // Filter logs by _user metadata field
  }) => {
    const params = new URLSearchParams()
    if (options?.workspace_id) params.set('workspace_id', options.workspace_id)
    if (options?.hours) params.set('hours', String(options.hours))
    if (options?.limit) params.set('limit', String(options.limit))
    if (options?.refresh) params.set('refresh', 'true')
    if (options?.user_filter) params.set('user_filter', options.user_filter)
    
    const { data } = await api.get<PortkeyLogsResponse>(`/logs/portkey/logs?${params}`)
    return data
  },
  
  getUsers: async () => {
    const { data } = await api.get<string[]>('/logs/portkey/users')
    return data
  },
  
  getLogById: async (logId: string) => {
    const { data } = await api.get<PortkeyLog>(`/logs/portkey/logs/${logId}`)
    return data
  },
  
  listExports: async (workspaceId?: string, limit = 20) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (workspaceId) params.set('workspace_id', workspaceId)
    const { data } = await api.get(`/logs/portkey/exports?${params}`)
    return data
  },
  
  createExport: async (options?: { workspace_id?: string; start_date?: string; end_date?: string }) => {
    const params = new URLSearchParams()
    if (options?.workspace_id) params.set('workspace_id', options.workspace_id)
    if (options?.start_date) params.set('start_date', options.start_date)
    if (options?.end_date) params.set('end_date', options.end_date)
    const { data } = await api.post(`/logs/portkey/exports/create?${params}`)
    return data
  },
  
  getExportStatus: async (exportId: string) => {
    const { data } = await api.get(`/logs/portkey/exports/${exportId}/status`)
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

// Evaluations - Replay-based evaluation pipeline
export interface AnalysisReport {
  analysis_id: string
  project_id: string
  log_count: number
  date_range: { start: string | null; end: string | null }
  token_distribution: Record<string, unknown>
  latency_metrics: {
    min_ms: number
    max_ms: number
    mean_ms: number
    std_ms: number
    p50_ms: number
    p75_ms: number
    p90_ms: number
    p95_ms: number
    p99_ms: number
  }
  cost_breakdown: {
    total_cost_usd: number
    avg_cost_per_request: number
    cost_by_model: Record<string, number>
    cost_by_provider: Record<string, number>
    projected_monthly_cost: number
  }
  prompt_complexity: Record<string, unknown>
  error_patterns: {
    total_requests: number
    success_rate: number
    error_rate: number
    refusal_rate: number
    timeout_rate: number
    error_codes: Record<string, number>
  }
  model_performance: Array<{
    model: string
    provider: string
    request_count: number
    avg_latency_ms: number
    p95_latency_ms: number
    avg_cost_per_request: number
    success_rate: number
    refusal_rate: number
    avg_input_tokens: number
    avg_output_tokens: number
  }>
  key_insights: string[]
  criteria_assessment: Record<string, unknown>
}

export interface CandidateModel {
  provider: string
  model: string
  rank: number
  reasoning: string
  expected_cost_per_request: number
  expected_latency_ms: number
  strengths: string[]
  concerns: string[]
  tier: string | null
  use_cases: string[]
  quality_score: number | null
  speed_score: number | null
}

export interface ModelSelectionResult {
  selection_id: string
  project_id: string
  candidates: CandidateModel[]
  selection_reasoning: string
  key_requirements: string[]
  confidence: number
}

export interface ReplayStatus {
  evaluation_run_id: string
  status: string
  progress: { completed: number; total: number }
  models_status: Array<{
    provider: string
    model: string
    status: string
    successful?: number
    failed?: number
    cost_usd?: number
  }>
}

export interface ModelResultSummary {
  provider: string
  model: string
  total_evaluated: number
  avg_quality_score: number
  avg_comparison_score: number
  comparison_verdicts: { better: number; equivalent: number; worse: number; unknown: number }
  total_cost_usd: number
  avg_latency_ms: number
  p95_latency_ms: number
  quality_distribution: { low: number; medium: number; high: number }
}

export interface EvaluationResults {
  evaluation_run_id: string
  project_id: string
  status: string
  completed_at: string | null
  total_logs_evaluated: number
  model_results: ModelResultSummary[]
  recommended_model: string | null
  recommended_provider: string | null
  recommendation_reasoning: string
  recommendation_confidence: number
}

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
  
  // New replay-based evaluation endpoints
  analyzeLogs: async (projectId: string, logIds: string[]) => {
    const { data } = await api.post<AnalysisReport>(`/evaluations/${projectId}/analyze`, {
      log_ids: logIds,
    })
    return data
  },
  
  selectModels: async (projectId: string, logIds?: string[], maxCandidates = 5) => {
    const { data } = await api.post<ModelSelectionResult>(`/evaluations/${projectId}/select-models`, {
      log_ids: logIds,
      max_candidates: maxCandidates,
    })
    return data
  },
  
  startReplay: async (
    projectId: string, 
    logIds: string[], 
    candidateModels: Array<{ provider: string; model: string }>,
    includeCurrentModel = true
  ) => {
    const { data } = await api.post<ReplayStatus>(`/evaluations/${projectId}/start-replay`, {
      log_ids: logIds,
      candidate_models: candidateModels,
      include_current_model: includeCurrentModel,
    })
    return data
  },
  
  getStatus: async (projectId: string, evaluationId: string) => {
    const { data } = await api.get<ReplayStatus>(`/evaluations/${projectId}/${evaluationId}/status`)
    return data
  },
  
  getResults: async (projectId: string, evaluationId: string) => {
    const { data } = await api.get<EvaluationResults>(`/evaluations/${projectId}/${evaluationId}/results`)
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
