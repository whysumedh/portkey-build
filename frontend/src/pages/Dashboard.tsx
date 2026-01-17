import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { 
  Activity, 
  DollarSign, 
  Clock, 
  AlertTriangle,
  ArrowRight,
  Sparkles
} from 'lucide-react'
import Card, { MetricCard, CardHeader } from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import { projectsApi } from '../api/client'
import type { Project } from '../types'

export default function Dashboard() {
  const { data: projectsData, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: () => projectsApi.list(1, 10, true),
  })

  const projects = projectsData?.items || []

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-void-50">Dashboard</h1>
        <p className="text-void-400 mt-1">Monitor your AI agents and model recommendations</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Active Projects"
          value={projects.length}
          icon={<Activity className="w-6 h-6" />}
        />
        <MetricCard
          label="Total Cost (30d)"
          value="$--"
          change={-12.5}
          changeLabel="vs last month"
          icon={<DollarSign className="w-6 h-6" />}
        />
        <MetricCard
          label="Avg Latency"
          value="--ms"
          change={-8.2}
          changeLabel="improvement"
          icon={<Clock className="w-6 h-6" />}
        />
        <MetricCard
          label="Pending Reviews"
          value={0}
          icon={<AlertTriangle className="w-6 h-6" />}
        />
      </div>

      {/* Projects Overview */}
      <Card>
        <CardHeader 
          title="Your Projects" 
          subtitle="AI agents being optimized"
          action={
            <Link to="/projects/new">
              <Button variant="primary" size="sm">
                <Sparkles className="w-4 h-4" />
                New Project
              </Button>
            </Link>
          }
        />
        
        {isLoading ? (
          <div className="text-center py-12 text-void-400">
            <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full mx-auto mb-4" />
            Loading projects...
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-void-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-8 h-8 text-void-500" />
            </div>
            <h3 className="text-lg font-medium text-void-200 mb-2">No projects yet</h3>
            <p className="text-void-400 mb-6">Create your first project to start optimizing your AI agents</p>
            <Link to="/projects/new">
              <Button variant="primary">Create Project</Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {projects.map((project) => (
              <ProjectRow key={project.id} project={project} />
            ))}
            
            {projects.length > 5 && (
              <div className="pt-4 border-t border-void-700/50">
                <Link to="/projects" className="text-accent-400 hover:text-accent-300 text-sm font-medium flex items-center gap-1">
                  View all projects <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader title="Recent Activity" subtitle="Latest evaluations and recommendations" />
        <div className="text-center py-8 text-void-400">
          <p>No recent activity to display</p>
        </div>
      </Card>
    </div>
  )
}

function ProjectRow({ project }: { project: Project }) {
  return (
    <Link 
      to={`/projects/${project.id}`}
      className="flex items-center justify-between p-4 rounded-lg bg-void-800/50 hover:bg-void-800 transition-colors group"
    >
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-lg bg-void-700 flex items-center justify-center">
          <span className="text-lg font-bold text-void-300">
            {project.name.charAt(0).toUpperCase()}
          </span>
        </div>
        <div>
          <h4 className="font-medium text-void-100 group-hover:text-white transition-colors">
            {project.name}
          </h4>
          <p className="text-sm text-void-400">
            {project.current_model ? `${project.current_provider}/${project.current_model}` : 'No model configured'}
          </p>
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <Badge variant={project.is_active ? 'success' : 'default'}>
          {project.is_active ? 'Active' : 'Inactive'}
        </Badge>
        <ArrowRight className="w-5 h-5 text-void-500 group-hover:text-void-300 transition-colors" />
      </div>
    </Link>
  )
}
