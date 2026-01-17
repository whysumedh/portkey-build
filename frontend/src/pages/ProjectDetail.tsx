import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  ArrowLeft, 
  RefreshCw, 
  BarChart3, 
  Sparkles, 
  Clock, 
  DollarSign,
  AlertCircle,
  CheckCircle2,
  Play
} from 'lucide-react'
import Card, { CardHeader, MetricCard } from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import { projectsApi, logsApi, recommendationsApi, schedulerApi } from '../api/client'
import { formatDistanceToNow, format } from 'date-fns'

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>()
  const queryClient = useQueryClient()

  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.get(id!),
    enabled: !!id,
  })

  const { data: stats } = useQuery({
    queryKey: ['project-stats', id],
    queryFn: () => logsApi.getStats(id!),
    enabled: !!id,
  })

  const { data: latestRec } = useQuery({
    queryKey: ['recommendation-latest', id],
    queryFn: () => recommendationsApi.getLatest(id!).catch(() => null),
    enabled: !!id,
  })

  const syncMutation = useMutation({
    mutationFn: () => logsApi.sync(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project-stats', id] })
      queryClient.invalidateQueries({ queryKey: ['project', id] })
    },
  })

  const triggerEvalMutation = useMutation({
    mutationFn: () => schedulerApi.triggerEvaluation(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', id] })
    },
  })

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!project) {
    return (
      <div className="text-center py-20">
        <AlertCircle className="w-16 h-16 text-void-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-void-200 mb-2">Project Not Found</h2>
        <Link to="/projects">
          <Button variant="secondary">Back to Projects</Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link 
            to="/projects" 
            className="flex items-center gap-2 text-void-400 hover:text-void-200 mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Projects
          </Link>
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-void-700 to-void-800 flex items-center justify-center">
              <span className="text-2xl font-bold text-void-200">
                {project.name.charAt(0).toUpperCase()}
              </span>
            </div>
            <div>
              <h1 className="text-2xl font-display font-bold text-void-50">{project.name}</h1>
              <p className="text-void-400">{project.agent_purpose}</p>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Button 
            variant="secondary"
            onClick={() => syncMutation.mutate()}
            loading={syncMutation.isPending}
          >
            <RefreshCw className="w-4 h-4" />
            Sync Logs
          </Button>
          <Button 
            variant="primary"
            onClick={() => triggerEvalMutation.mutate()}
            loading={triggerEvalMutation.isPending}
          >
            <Play className="w-4 h-4" />
            Run Evaluation
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Requests"
          value={stats?.total_logs?.toLocaleString() || '0'}
          icon={<BarChart3 className="w-6 h-6" />}
        />
        <MetricCard
          label="Total Cost"
          value={`$${stats?.total_cost_usd?.toFixed(2) || '0.00'}`}
          icon={<DollarSign className="w-6 h-6" />}
        />
        <MetricCard
          label="Avg Latency"
          value={`${stats?.avg_latency_ms?.toFixed(0) || '0'}ms`}
          icon={<Clock className="w-6 h-6" />}
        />
        <MetricCard
          label="Success Rate"
          value={`${((stats?.success_rate || 0) * 100).toFixed(1)}%`}
          icon={<CheckCircle2 className="w-6 h-6" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Configuration */}
        <Card>
          <CardHeader 
            title="Current Configuration" 
            subtitle="Model and criteria settings"
          />
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-void-800/50 rounded-lg">
              <span className="text-void-400">Current Model</span>
              <span className="font-mono text-void-100">
                {project.current_model 
                  ? `${project.current_provider}/${project.current_model}`
                  : 'Not configured'
                }
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-void-800/50 rounded-lg">
              <span className="text-void-400">Status</span>
              <Badge variant={project.is_active ? 'success' : 'default'}>
                {project.is_active ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-void-800/50 rounded-lg">
              <span className="text-void-400">Last Sync</span>
              <span className="text-void-200">
                {project.last_log_sync 
                  ? formatDistanceToNow(new Date(project.last_log_sync), { addSuffix: true })
                  : 'Never'
                }
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-void-800/50 rounded-lg">
              <span className="text-void-400">Last Evaluation</span>
              <span className="text-void-200">
                {project.last_evaluation 
                  ? formatDistanceToNow(new Date(project.last_evaluation), { addSuffix: true })
                  : 'Never'
                }
              </span>
            </div>
          </div>
        </Card>

        {/* Latest Recommendation */}
        <Card>
          <CardHeader 
            title="Latest Recommendation" 
            subtitle="Model optimization suggestion"
            action={
              <Link to={`/projects/${id}/recommendations`}>
                <Button variant="ghost" size="sm">View All</Button>
              </Link>
            }
          />
          
          {latestRec ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  latestRec.status === 'recommended' 
                    ? 'bg-success-500/20 text-success-400'
                    : 'bg-void-700 text-void-400'
                }`}>
                  <Sparkles className="w-5 h-5" />
                </div>
                <div>
                  <p className="font-medium text-void-100">
                    {latestRec.status === 'recommended' 
                      ? `Switch to ${latestRec.recommended_provider}/${latestRec.recommended_model}`
                      : 'No recommendation'
                    }
                  </p>
                  <p className="text-sm text-void-400">
                    {latestRec.confidence_score * 100}% confidence
                  </p>
                </div>
              </div>
              
              {latestRec.key_insights?.slice(0, 2).map((insight, i) => (
                <p key={i} className="text-sm text-void-300 pl-4 border-l-2 border-void-700">
                  {insight}
                </p>
              ))}
              
              <p className="text-xs text-void-500">
                Generated {format(new Date(latestRec.created_at), 'PPP')}
              </p>
            </div>
          ) : (
            <div className="text-center py-8">
              <Sparkles className="w-12 h-12 text-void-600 mx-auto mb-3" />
              <p className="text-void-400 mb-4">No recommendations yet</p>
              <Button variant="secondary" size="sm" onClick={() => triggerEvalMutation.mutate()}>
                Generate Recommendation
              </Button>
            </div>
          )}
        </Card>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link to={`/projects/${id}/analytics`}>
          <Card hover className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h4 className="font-medium text-void-100">View Analytics</h4>
              <p className="text-sm text-void-400">Explore detailed metrics</p>
            </div>
          </Card>
        </Link>
        
        <Link to={`/projects/${id}/recommendations`}>
          <Card hover className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-accent-500/20 flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-accent-400" />
            </div>
            <div>
              <h4 className="font-medium text-void-100">Recommendations</h4>
              <p className="text-sm text-void-400">View all suggestions</p>
            </div>
          </Card>
        </Link>
        
        <Card hover className="flex items-center gap-4 opacity-60 cursor-not-allowed">
          <div className="w-12 h-12 rounded-xl bg-void-700 flex items-center justify-center">
            <Clock className="w-6 h-6 text-void-400" />
          </div>
          <div>
            <h4 className="font-medium text-void-100">Evaluation History</h4>
            <p className="text-sm text-void-400">Coming soon</p>
          </div>
        </Card>
      </div>
    </div>
  )
}
