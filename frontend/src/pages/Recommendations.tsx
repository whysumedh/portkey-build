import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Sparkles, AlertTriangle, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import Card, { CardHeader } from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import { recommendationsApi, projectsApi } from '../api/client'
import { format } from 'date-fns'
import type { Recommendation, CandidateModel } from '../types'
import clsx from 'clsx'

export default function Recommendations() {
  const { id } = useParams<{ id: string }>()

  const { data: project } = useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.get(id!),
    enabled: !!id,
  })

  const { data: recsData, isLoading } = useQuery({
    queryKey: ['recommendations', id],
    queryFn: () => recommendationsApi.list(id!),
    enabled: !!id,
  })

  const recommendations = recsData?.items || []

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <Link 
          to={`/projects/${id}`}
          className="flex items-center gap-2 text-void-400 hover:text-void-200 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to {project?.name || 'Project'}
        </Link>
        <h1 className="text-2xl font-display font-bold text-void-50">Recommendations</h1>
        <p className="text-void-400 mt-1">Model optimization suggestions for {project?.name}</p>
      </div>

      {/* Recommendations List */}
      {isLoading ? (
        <div className="text-center py-20 text-void-400">
          <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full mx-auto mb-4" />
          Loading recommendations...
        </div>
      ) : recommendations.length === 0 ? (
        <Card>
          <div className="text-center py-16">
            <Sparkles className="w-16 h-16 text-void-600 mx-auto mb-4" />
            <h3 className="text-xl font-medium text-void-200 mb-2">No recommendations yet</h3>
            <p className="text-void-400 mb-6">Run an evaluation to generate model recommendations</p>
            <Link to={`/projects/${id}`}>
              <Button variant="primary">Go to Project</Button>
            </Link>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {recommendations.map((rec) => (
            <RecommendationCard key={rec.id} recommendation={rec} projectId={id!} />
          ))}
        </div>
      )}
    </div>
  )
}

function RecommendationCard({ recommendation, projectId }: { recommendation: Recommendation; projectId: string }) {
  const [expanded, setExpanded] = useState(false)

  const statusConfig = {
    recommended: { icon: CheckCircle, color: 'text-success-400', bg: 'bg-success-500/20', label: 'Recommended' },
    no_recommendation: { icon: XCircle, color: 'text-void-400', bg: 'bg-void-700', label: 'No Recommendation' },
    low_confidence: { icon: AlertTriangle, color: 'text-warning-400', bg: 'bg-warning-500/20', label: 'Low Confidence' },
    insufficient_data: { icon: AlertTriangle, color: 'text-void-400', bg: 'bg-void-700', label: 'Insufficient Data' },
  }

  const config = statusConfig[recommendation.status] || statusConfig.no_recommendation
  const StatusIcon = config.icon

  return (
    <Card>
      <div 
        className="flex items-start justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-4">
          <div className={clsx('w-12 h-12 rounded-xl flex items-center justify-center', config.bg)}>
            <StatusIcon className={clsx('w-6 h-6', config.color)} />
          </div>
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h3 className="font-semibold text-void-100">
                {recommendation.status === 'recommended' 
                  ? `Switch to ${recommendation.recommended_provider}/${recommendation.recommended_model}`
                  : config.label
                }
              </h3>
              <Badge variant={recommendation.acknowledged ? 'default' : 'warning'}>
                {recommendation.acknowledged ? 'Acknowledged' : 'Pending Review'}
              </Badge>
            </div>
            <p className="text-sm text-void-400">
              {format(new Date(recommendation.created_at), 'PPP p')}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-void-400">Confidence</p>
            <p className={clsx(
              'text-lg font-bold',
              recommendation.confidence_score >= 0.7 ? 'text-success-400' :
              recommendation.confidence_score >= 0.5 ? 'text-warning-400' : 'text-void-400'
            )}>
              {(recommendation.confidence_score * 100).toFixed(0)}%
            </p>
          </div>
          {expanded ? <ChevronUp className="w-5 h-5 text-void-500" /> : <ChevronDown className="w-5 h-5 text-void-500" />}
        </div>
      </div>

      {expanded && (
        <div className="mt-6 pt-6 border-t border-void-700/50 space-y-6">
          {/* Reasoning */}
          {recommendation.reasoning && (
            <div>
              <h4 className="font-medium text-void-200 mb-2">Reasoning</h4>
              <p className="text-void-300">{recommendation.reasoning}</p>
            </div>
          )}

          {/* Key Insights */}
          {recommendation.key_insights?.length > 0 && (
            <div>
              <h4 className="font-medium text-void-200 mb-3">Key Insights</h4>
              <ul className="space-y-2">
                {recommendation.key_insights.map((insight, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Sparkles className="w-4 h-4 text-accent-400 mt-0.5 flex-shrink-0" />
                    <span className="text-void-300">{insight}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Risk Notes */}
          {recommendation.risk_notes?.length > 0 && (
            <div>
              <h4 className="font-medium text-void-200 mb-3">Risk Notes</h4>
              <ul className="space-y-2">
                {recommendation.risk_notes.map((note, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-warning-400 mt-0.5 flex-shrink-0" />
                    <span className="text-void-300">{note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Candidates */}
          {recommendation.candidates?.length > 0 && (
            <div>
              <h4 className="font-medium text-void-200 mb-3">Candidate Models</h4>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-void-400 border-b border-void-700">
                      <th className="pb-2 pr-4">Rank</th>
                      <th className="pb-2 pr-4">Model</th>
                      <th className="pb-2 pr-4">Quality</th>
                      <th className="pb-2 pr-4">Latency</th>
                      <th className="pb-2 pr-4">Cost</th>
                      <th className="pb-2">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendation.candidates.map((candidate) => (
                      <CandidateRow key={candidate.id} candidate={candidate} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Actions */}
          {!recommendation.acknowledged && (
            <div className="flex justify-end gap-3 pt-4 border-t border-void-700/50">
              <Button variant="ghost">Dismiss</Button>
              <Button variant="primary">Acknowledge</Button>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

function CandidateRow({ candidate }: { candidate: CandidateModel }) {
  return (
    <tr className="border-b border-void-800 text-sm">
      <td className="py-3 pr-4">
        <span className={clsx(
          'w-6 h-6 rounded-full inline-flex items-center justify-center text-xs font-bold',
          candidate.rank === 1 ? 'bg-accent-500/20 text-accent-400' : 'bg-void-700 text-void-400'
        )}>
          {candidate.rank}
        </span>
      </td>
      <td className="py-3 pr-4">
        <span className="font-mono text-void-100">
          {candidate.provider}/{candidate.model}
        </span>
        {candidate.is_current && (
          <Badge variant="info" size="sm" className="ml-2">Current</Badge>
        )}
      </td>
      <td className="py-3 pr-4">
        <span className={clsx(
          candidate.quality_score >= 0.8 ? 'text-success-400' :
          candidate.quality_score >= 0.6 ? 'text-warning-400' : 'text-void-400'
        )}>
          {(candidate.quality_score * 100).toFixed(0)}%
        </span>
      </td>
      <td className="py-3 pr-4 text-void-300">
        {candidate.avg_latency_ms.toFixed(0)}ms
      </td>
      <td className="py-3 pr-4 text-void-300">
        ${candidate.avg_cost_per_request.toFixed(4)}
      </td>
      <td className="py-3">
        <span className="font-medium text-void-100">
          {candidate.overall_score.toFixed(3)}
        </span>
      </td>
    </tr>
  )
}
