import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { ArrowLeft, Sparkles } from 'lucide-react'
import Card from '../components/Card'
import Button from '../components/Button'
import { projectsApi } from '../api/client'

export default function ProjectCreate() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    name: '',
    agent_purpose: '',
    description: '',
    portkey_virtual_key: '',
    current_model: '',
    current_provider: '',
  })

  const createMutation = useMutation({
    mutationFn: (data: typeof formData) => projectsApi.create(data),
    onSuccess: (project) => {
      navigate(`/projects/${project.id}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate(formData)
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <button 
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-void-400 hover:text-void-200 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <h1 className="text-2xl font-display font-bold text-void-50">Create New Project</h1>
        <p className="text-void-400 mt-1">Set up a new AI agent for model optimization</p>
      </div>

      {/* Form */}
      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Info */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-accent-500" />
              Basic Information
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Project Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Customer Support Bot"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Agent Purpose *
                </label>
                <textarea
                  value={formData.agent_purpose}
                  onChange={(e) => setFormData({ ...formData, agent_purpose: e.target.value })}
                  placeholder="Describe what this AI agent does and its primary use cases..."
                  rows={3}
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500 resize-none"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Additional notes about this project..."
                  rows={2}
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500 resize-none"
                />
              </div>
            </div>
          </div>

          {/* Portkey Integration */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4">Portkey Integration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Portkey Virtual Key
                </label>
                <input
                  type="text"
                  value={formData.portkey_virtual_key}
                  onChange={(e) => setFormData({ ...formData, portkey_virtual_key: e.target.value })}
                  placeholder="pk-..."
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500 font-mono"
                />
                <p className="text-xs text-void-500 mt-1">
                  Used to filter and sync logs from Portkey
                </p>
              </div>
            </div>
          </div>

          {/* Current Model */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4">Current Model Configuration</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Provider
                </label>
                <input
                  type="text"
                  value={formData.current_provider}
                  onChange={(e) => setFormData({ ...formData, current_provider: e.target.value })}
                  placeholder="e.g., openai"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Model
                </label>
                <input
                  type="text"
                  value={formData.current_model}
                  onChange={(e) => setFormData({ ...formData, current_model: e.target.value })}
                  placeholder="e.g., gpt-4o"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                />
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-void-700">
            <Button 
              type="button" 
              variant="ghost"
              onClick={() => navigate(-1)}
            >
              Cancel
            </Button>
            <Button 
              type="submit" 
              variant="primary"
              loading={createMutation.isPending}
            >
              Create Project
            </Button>
          </div>

          {createMutation.isError && (
            <div className="p-4 bg-accent-500/10 border border-accent-500/30 rounded-lg text-accent-400 text-sm">
              Failed to create project. Please try again.
            </div>
          )}
        </form>
      </Card>
    </div>
  )
}
