import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Plus, Search, Filter, MoreVertical } from 'lucide-react'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import { projectsApi } from '../api/client'
import { formatDistanceToNow } from 'date-fns'

export default function ProjectList() {
  const { data, isLoading } = useQuery({
    queryKey: ['projects', 'all'],
    queryFn: () => projectsApi.list(1, 50),
  })

  const projects = data?.items || []

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-void-50">Projects</h1>
          <p className="text-void-400 mt-1">Manage your AI agent optimization projects</p>
        </div>
        <Link to="/projects/new">
          <Button variant="primary">
            <Plus className="w-4 h-4" />
            New Project
          </Button>
        </Link>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-void-500" />
          <input
            type="text"
            placeholder="Search projects..."
            className="w-full pl-10 pr-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
          />
        </div>
        <Button variant="secondary">
          <Filter className="w-4 h-4" />
          Filters
        </Button>
      </div>

      {/* Projects Grid */}
      {isLoading ? (
        <div className="text-center py-20 text-void-400">
          <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full mx-auto mb-4" />
          Loading projects...
        </div>
      ) : projects.length === 0 ? (
        <Card>
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-void-800 rounded-full flex items-center justify-center mx-auto mb-6">
              <Plus className="w-10 h-10 text-void-500" />
            </div>
            <h3 className="text-xl font-medium text-void-200 mb-2">No projects yet</h3>
            <p className="text-void-400 mb-8 max-w-md mx-auto">
              Create your first project to start analyzing your AI agent's performance and get model recommendations.
            </p>
            <Link to="/projects/new">
              <Button variant="primary" size="lg">Create Your First Project</Button>
            </Link>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map((project) => (
            <Link key={project.id} to={`/projects/${project.id}`}>
              <Card hover className="h-full">
                <div className="flex items-start justify-between mb-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-void-700 to-void-800 flex items-center justify-center">
                    <span className="text-xl font-bold text-void-300">
                      {project.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <button 
                    className="p-1 rounded hover:bg-void-700 text-void-500 hover:text-void-300"
                    onClick={(e) => e.preventDefault()}
                  >
                    <MoreVertical className="w-5 h-5" />
                  </button>
                </div>
                
                <h3 className="font-semibold text-void-100 mb-1">{project.name}</h3>
                <p className="text-sm text-void-400 line-clamp-2 mb-4">
                  {project.agent_purpose || 'No description'}
                </p>
                
                <div className="flex items-center justify-between pt-4 border-t border-void-700/50">
                  <Badge variant={project.is_active ? 'success' : 'default'}>
                    {project.is_active ? 'Active' : 'Inactive'}
                  </Badge>
                  <span className="text-xs text-void-500">
                    {project.last_evaluation 
                      ? `Evaluated ${formatDistanceToNow(new Date(project.last_evaluation), { addSuffix: true })}`
                      : 'Never evaluated'
                    }
                  </span>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
