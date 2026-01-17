// Project types
export interface SuccessCriteria {
  id: string
  project_id: string
  min_accuracy: number
  min_quality_score: number
  max_latency_ms: number
  max_latency_p95_ms: number
  max_cost_per_request_usd: number
  max_monthly_cost_usd?: number
  max_refusal_rate: number
  max_safety_violations: number
  safety_categories_blocked: string[]
  created_at: string
  updated_at: string
}

export interface ToleranceLevels {
  id: string
  project_id: string
  cost_sensitivity: 'low' | 'medium' | 'high'
  latency_tolerance_pct: number
  cost_tolerance_pct: number
  quality_tolerance_pct: number
  absolute_max_latency_ms: number
  absolute_max_refusal_rate: number
  created_at: string
  updated_at: string
}

export interface Project {
  id: string
  name: string
  description?: string
  agent_purpose: string
  portkey_virtual_key?: string
  portkey_config_id?: string
  current_model?: string
  current_provider?: string
  selected_log_ids?: string[]
  log_filter_metadata?: Record<string, string>
  is_active: boolean
  last_log_sync?: string
  last_evaluation?: string
  version: number
  created_at: string
  updated_at: string
  success_criteria?: SuccessCriteria
  tolerance_levels?: ToleranceLevels
}

// Log types
export interface LogEntry {
  id: string
  project_id: string
  portkey_log_id: string
  timestamp: string
  model: string
  provider: string
  input_tokens: number
  output_tokens: number
  latency_ms: number
  cost_usd: number
  status: string
  refusal: boolean
}

export interface LogStats {
  project_id: string
  total_logs: number
  date_range_start?: string
  date_range_end?: string
  models_used: string[]
  providers_used: string[]
  total_cost_usd: number
  total_input_tokens: number
  total_output_tokens: number
  avg_latency_ms: number
  success_rate: number
  refusal_rate: number
  error_rate: number
}

// Recommendation types
export interface CandidateModel {
  id: string
  model: string
  provider: string
  is_current: boolean
  rank: number
  avg_latency_ms: number
  p95_latency_ms: number
  avg_cost_per_request: number
  success_rate: number
  refusal_rate: number
  quality_score: number
  correctness_score: number
  helpfulness_score: number
  safety_score: number
  overall_score: number
  trade_offs?: Record<string, unknown>
  selection_reason?: string
}

export interface Recommendation {
  id: string
  project_id: string
  evaluation_run_id?: string
  status: 'recommended' | 'no_recommendation' | 'low_confidence' | 'insufficient_data'
  recommended_model?: string
  recommended_provider?: string
  current_model?: string
  current_provider?: string
  confidence_score: number
  risk_level: string
  risk_notes: string[]
  trade_off_summary?: Record<string, unknown>
  reasoning?: string
  key_insights: string[]
  example_improvements: unknown[]
  example_regressions: unknown[]
  judge_disagreements: unknown[]
  version: number
  created_at: string
  acknowledged: boolean
  acknowledged_at?: string
  acknowledged_by?: string
  action_taken?: string
  candidates: CandidateModel[]
}

// Evaluation types
export interface EvaluationRun {
  id: string
  project_id: string
  trigger_type: string
  sample_size: number
  candidate_models: string[]
  logs_start_date: string
  logs_end_date: string
  total_logs_analyzed: number
  status: string
  error_message?: string
  replays_completed: number
  replays_total: number
  judgments_completed: number
  judgments_total: number
  started_at?: string
  completed_at?: string
  version: number
  created_at: string
}

// Analytics types
export interface AnalysisSummary {
  total_logs: number
  date_range: {
    start?: string
    end?: string
  }
  models: string[]
  providers: string[]
  status_distribution: Record<string, number>
  refusal_rate: number
  numeric_summaries: Record<string, {
    count: number
    mean: number
    std: number
    min: number
    p25: number
    p50: number
    p75: number
    p95: number
    max: number
  }>
}

// Pagination types
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  pages: number
}
