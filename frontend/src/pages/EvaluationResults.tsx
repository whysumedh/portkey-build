import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { 
  ArrowLeft,
  DollarSign, 
  Clock, 
  Star,
  CheckCircle,
  XCircle,
  MinusCircle,
  Trophy,
  BarChart3,
  Activity,
  RefreshCw,
} from 'lucide-react'
import Card, { CardHeader, MetricCard } from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import { evaluationsApi } from '../api/client'
import type { EvaluationResults as EvaluationResultsType, ModelResultSummary } from '../api/client'

export default function EvaluationResults() {
  const { projectId, evaluationId } = useParams<{ projectId: string; evaluationId: string }>()
  const [pollingEnabled, setPollingEnabled] = useState(true)
  
  // Query for evaluation status
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['evaluation-status', projectId, evaluationId],
    queryFn: () => evaluationsApi.getStatus(projectId!, evaluationId!),
    enabled: !!projectId && !!evaluationId && pollingEnabled,
    refetchInterval: pollingEnabled ? 3000 : false,
  })
  
  // Query for full results (only when completed)
  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ['evaluation-results', projectId, evaluationId],
    queryFn: () => evaluationsApi.getResults(projectId!, evaluationId!),
    enabled: !!projectId && !!evaluationId && status?.status === 'completed',
  })
  
  // Stop polling when complete
  useEffect(() => {
    if (status?.status === 'completed' || status?.status === 'failed') {
      setPollingEnabled(false)
    }
  }, [status?.status])
  
  if (statusLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full" />
      </div>
    )
  }
  
  const isRunning = status?.status === 'running' || status?.status === 'pending'
  const isComplete = status?.status === 'completed'
  const isFailed = status?.status === 'failed'
  
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to={`/projects/${projectId}`}>
          <Button variant="ghost" size="sm">
            <ArrowLeft className="w-4 h-4" />
            Back to Project
          </Button>
        </Link>
      </div>
      
      <div>
        <h1 className="text-3xl font-display font-bold text-void-50">Evaluation Results</h1>
        <p className="text-void-400 mt-1">
          {isRunning ? 'Evaluation in progress...' : 
           isComplete ? 'Model comparison and recommendations' :
           'Evaluation failed'}
        </p>
      </div>
      
      {/* Status Banner */}
      {isRunning && (
        <Card className="border-l-4 border-l-accent-500 bg-accent-500/10">
          <div className="flex items-center gap-4">
            <div className="animate-spin">
              <RefreshCw className="w-6 h-6 text-accent-400" />
            </div>
            <div className="flex-1">
              <h3 className="font-medium text-void-100">Evaluation Running</h3>
              <p className="text-sm text-void-400">
                Replaying {status?.progress.total || 0} logs through candidate models...
              </p>
              <div className="mt-2 w-full bg-void-800 rounded-full h-2">
                <div 
                  className="bg-accent-500 h-2 rounded-full transition-all"
                  style={{ 
                    width: `${status?.progress.total ? (status.progress.completed / status.progress.total * 100) : 0}%` 
                  }}
                />
              </div>
              <p className="text-xs text-void-500 mt-1">
                {status?.progress.completed || 0} / {status?.progress.total || 0} completed
              </p>
            </div>
          </div>
          
          {/* Per-model status */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
            {status?.models_status.map((model, i) => (
              <div key={i} className="bg-void-800/50 rounded p-2">
                <div className="text-sm font-medium text-void-200 truncate">
                  {model.provider}/{model.model}
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <Badge variant={
                    model.status === 'completed' ? 'success' :
                    model.status === 'running' ? 'warning' : 'default'
                  } size="sm">
                    {model.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
      
      {isFailed && (
        <Card className="border-l-4 border-l-red-500 bg-red-500/10">
          <div className="flex items-center gap-4">
            <XCircle className="w-6 h-6 text-red-400" />
            <div>
              <h3 className="font-medium text-red-400">Evaluation Failed</h3>
              <p className="text-sm text-void-400">
                Something went wrong during the evaluation. Please try again.
              </p>
            </div>
          </div>
        </Card>
      )}
      
      {/* Results */}
      {isComplete && results && (
        <>
          {/* Recommendation Banner */}
          {results.recommended_model && (
            <Card className="border-l-4 border-l-emerald-500 bg-emerald-500/10">
              <div className="flex items-start gap-4">
                <Trophy className="w-8 h-8 text-emerald-400 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-emerald-400 text-lg">Recommended Model</h3>
                  <p className="text-xl font-bold text-void-50 mt-1">
                    {results.recommended_provider}/{results.recommended_model}
                  </p>
                  <p className="text-void-300 mt-2">{results.recommendation_reasoning}</p>
                  <div className="flex items-center gap-2 mt-3">
                    <Badge variant="success">
                      {Math.round(results.recommendation_confidence * 100)}% confidence
                    </Badge>
                  </div>
                </div>
              </div>
            </Card>
          )}
          
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard
              label="Models Tested"
              value={results.model_results.length}
              icon={<BarChart3 className="w-6 h-6" />}
            />
            <MetricCard
              label="Logs Evaluated"
              value={results.total_logs_evaluated}
              icon={<Activity className="w-6 h-6" />}
            />
            <MetricCard
              label="Total Cost"
              value={`$${results.model_results.reduce((sum, m) => sum + m.total_cost_usd, 0).toFixed(4)}`}
              icon={<DollarSign className="w-6 h-6" />}
            />
            <MetricCard
              label="Best Quality Score"
              value={`${Math.round(Math.max(...results.model_results.map(m => m.avg_quality_score)) * 100)}%`}
              icon={<Star className="w-6 h-6" />}
            />
          </div>
          
          {/* Model Comparison Table */}
          <Card>
            <CardHeader title="Model Comparison" subtitle="Performance metrics for each candidate" />
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-void-700">
                    <th className="text-left py-3 px-4 text-void-400 font-medium">Model</th>
                    <th className="text-right py-3 px-4 text-void-400 font-medium">Quality</th>
                    <th className="text-right py-3 px-4 text-void-400 font-medium">Comparison</th>
                    <th className="text-right py-3 px-4 text-void-400 font-medium">Cost</th>
                    <th className="text-right py-3 px-4 text-void-400 font-medium">Latency</th>
                    <th className="text-center py-3 px-4 text-void-400 font-medium">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {results.model_results
                    .sort((a, b) => b.avg_quality_score - a.avg_quality_score)
                    .map((model, i) => (
                      <ModelResultRow 
                        key={i} 
                        model={model} 
                        isRecommended={
                          model.model === results.recommended_model && 
                          model.provider === results.recommended_provider
                        }
                      />
                    ))}
                </tbody>
              </table>
            </div>
          </Card>
          
          {/* Cost Comparison Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader title="Cost Comparison" subtitle="Total cost per model" />
              <div className="space-y-3">
                {results.model_results
                  .sort((a, b) => a.total_cost_usd - b.total_cost_usd)
                  .map((model, i) => {
                    const maxCost = Math.max(...results.model_results.map(m => m.total_cost_usd))
                    const pct = maxCost > 0 ? (model.total_cost_usd / maxCost * 100) : 0
                    return (
                      <div key={i} className="flex items-center gap-3">
                        <div className="w-32 text-sm text-void-300 truncate">
                          {model.provider}/{model.model}
                        </div>
                        <div className="flex-1 bg-void-800 rounded-full h-4">
                          <div 
                            className="bg-emerald-500 h-4 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <div className="w-20 text-right text-sm text-void-200">
                          ${model.total_cost_usd.toFixed(4)}
                        </div>
                      </div>
                    )
                  })}
              </div>
            </Card>
            
            <Card>
              <CardHeader title="Latency Comparison" subtitle="Average response time per model" />
              <div className="space-y-3">
                {results.model_results
                  .sort((a, b) => a.avg_latency_ms - b.avg_latency_ms)
                  .map((model, i) => {
                    const maxLatency = Math.max(...results.model_results.map(m => m.avg_latency_ms))
                    const pct = maxLatency > 0 ? (model.avg_latency_ms / maxLatency * 100) : 0
                    return (
                      <div key={i} className="flex items-center gap-3">
                        <div className="w-32 text-sm text-void-300 truncate">
                          {model.provider}/{model.model}
                        </div>
                        <div className="flex-1 bg-void-800 rounded-full h-4">
                          <div 
                            className="bg-blue-500 h-4 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <div className="w-20 text-right text-sm text-void-200">
                          {model.avg_latency_ms.toFixed(0)}ms
                        </div>
                      </div>
                    )
                  })}
              </div>
            </Card>
          </div>
          
          {/* Quality Distribution */}
          <Card>
            <CardHeader title="Quality Distribution" subtitle="Response quality breakdown per model" />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {results.model_results.map((model, i) => (
                <div key={i} className="bg-void-800/50 rounded-lg p-4">
                  <h4 className="font-medium text-void-200 truncate mb-3">
                    {model.provider}/{model.model}
                  </h4>
                  <div className="space-y-2">
                    <QualityBar label="High" count={model.quality_distribution.high} total={model.total_evaluated} color="emerald" />
                    <QualityBar label="Medium" count={model.quality_distribution.medium} total={model.total_evaluated} color="amber" />
                    <QualityBar label="Low" count={model.quality_distribution.low} total={model.total_evaluated} color="red" />
                  </div>
                  
                  <div className="mt-4 pt-3 border-t border-void-700">
                    <div className="flex justify-between text-xs text-void-400">
                      <span>Comparison Verdicts:</span>
                    </div>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="success" size="sm">
                        <CheckCircle className="w-3 h-3 mr-1" />
                        {model.comparison_verdicts.better} better
                      </Badge>
                      <Badge variant="default" size="sm">
                        <MinusCircle className="w-3 h-3 mr-1" />
                        {model.comparison_verdicts.equivalent} eq
                      </Badge>
                      <Badge variant="error" size="sm">
                        <XCircle className="w-3 h-3 mr-1" />
                        {model.comparison_verdicts.worse} worse
                      </Badge>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

function ModelResultRow({ 
  model, 
  isRecommended 
}: { 
  model: ModelResultSummary
  isRecommended: boolean
}) {
  const betterPct = model.total_evaluated > 0 
    ? (model.comparison_verdicts.better / model.total_evaluated * 100) 
    : 0
  const equivPct = model.total_evaluated > 0 
    ? (model.comparison_verdicts.equivalent / model.total_evaluated * 100) 
    : 0
  const worsePct = model.total_evaluated > 0 
    ? (model.comparison_verdicts.worse / model.total_evaluated * 100) 
    : 0
  
  return (
    <tr className={`border-b border-void-800 hover:bg-void-800/50 ${isRecommended ? 'bg-emerald-500/5' : ''}`}>
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          {isRecommended && <Trophy className="w-4 h-4 text-emerald-400" />}
          <span className={`font-medium ${isRecommended ? 'text-emerald-400' : 'text-void-200'}`}>
            {model.provider}/{model.model}
          </span>
        </div>
      </td>
      <td className="py-3 px-4 text-right">
        <span className={`font-medium ${
          model.avg_quality_score >= 0.7 ? 'text-emerald-400' :
          model.avg_quality_score >= 0.5 ? 'text-amber-400' : 'text-red-400'
        }`}>
          {Math.round(model.avg_quality_score * 100)}%
        </span>
      </td>
      <td className="py-3 px-4 text-right">
        <span className="text-void-300">
          {Math.round(model.avg_comparison_score * 100)}%
        </span>
      </td>
      <td className="py-3 px-4 text-right text-void-300">
        ${model.total_cost_usd.toFixed(4)}
      </td>
      <td className="py-3 px-4 text-right text-void-300">
        {model.avg_latency_ms.toFixed(0)}ms
        <span className="text-void-500 text-xs ml-1">
          (p95: {model.p95_latency_ms.toFixed(0)}ms)
        </span>
      </td>
      <td className="py-3 px-4">
        <div className="flex justify-center gap-1">
          <span className="text-emerald-400 text-xs">{betterPct.toFixed(0)}%↑</span>
          <span className="text-void-500">/</span>
          <span className="text-void-400 text-xs">{equivPct.toFixed(0)}%=</span>
          <span className="text-void-500">/</span>
          <span className="text-red-400 text-xs">{worsePct.toFixed(0)}%↓</span>
        </div>
      </td>
    </tr>
  )
}

function QualityBar({ 
  label, 
  count, 
  total, 
  color 
}: { 
  label: string
  count: number
  total: number
  color: 'emerald' | 'amber' | 'red'
}) {
  const pct = total > 0 ? (count / total * 100) : 0
  const bgColor = {
    emerald: 'bg-emerald-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
  }[color]
  
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 text-xs text-void-400">{label}</span>
      <div className="flex-1 bg-void-700 rounded-full h-2">
        <div 
          className={`${bgColor} h-2 rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-12 text-right text-xs text-void-400">
        {count} ({pct.toFixed(0)}%)
      </span>
    </div>
  )
}
