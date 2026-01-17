import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, TrendingUp, Clock, DollarSign, AlertTriangle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts'
import Card, { CardHeader } from '../components/Card'
import { analyticsApi, projectsApi } from '../api/client'

const COLORS = ['#f43f5e', '#3b82f6', '#22c55e', '#eab308', '#8b5cf6']

export default function Analytics() {
  const { id } = useParams<{ id: string }>()

  const { data: project } = useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.get(id!),
    enabled: !!id,
  })

  const { data: summary, isLoading } = useQuery({
    queryKey: ['analytics-summary', id],
    queryFn: () => analyticsApi.getSummary(id!),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  // Prepare chart data
  const statusData = Object.entries(summary?.status_distribution || {}).map(([status, count]) => ({
    name: status,
    value: count as number,
  }))

  const latencyData = summary?.numeric_summaries?.latency_ms ? [
    { name: 'Min', value: summary.numeric_summaries.latency_ms.min },
    { name: 'P25', value: summary.numeric_summaries.latency_ms.p25 },
    { name: 'P50', value: summary.numeric_summaries.latency_ms.p50 },
    { name: 'P75', value: summary.numeric_summaries.latency_ms.p75 },
    { name: 'P95', value: summary.numeric_summaries.latency_ms.p95 },
    { name: 'Max', value: summary.numeric_summaries.latency_ms.max },
  ] : []

  const tokenData = summary?.numeric_summaries ? [
    { name: 'Input Tokens', avg: summary.numeric_summaries.input_tokens?.mean || 0 },
    { name: 'Output Tokens', avg: summary.numeric_summaries.output_tokens?.mean || 0 },
  ] : []

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
        <h1 className="text-2xl font-display font-bold text-void-50">Analytics</h1>
        <p className="text-void-400 mt-1">
          Analyzing {summary?.total_logs?.toLocaleString() || 0} requests
          {summary?.date_range?.start && summary?.date_range?.end && (
            <span> from {new Date(summary.date_range.start).toLocaleDateString()} to {new Date(summary.date_range.end).toLocaleDateString()}</span>
          )}
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <p className="text-sm text-void-400">Total Requests</p>
            <p className="text-xl font-bold text-void-100">{summary?.total_logs?.toLocaleString() || 0}</p>
          </div>
        </Card>
        
        <Card className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
            <Clock className="w-6 h-6 text-success-400" />
          </div>
          <div>
            <p className="text-sm text-void-400">Avg Latency</p>
            <p className="text-xl font-bold text-void-100">
              {summary?.numeric_summaries?.latency_ms?.mean?.toFixed(0) || 0}ms
            </p>
          </div>
        </Card>
        
        <Card className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
            <DollarSign className="w-6 h-6 text-warning-400" />
          </div>
          <div>
            <p className="text-sm text-void-400">Avg Cost</p>
            <p className="text-xl font-bold text-void-100">
              ${summary?.numeric_summaries?.cost_usd?.mean?.toFixed(4) || '0.00'}
            </p>
          </div>
        </Card>
        
        <Card className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-accent-500/20 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6 text-accent-400" />
          </div>
          <div>
            <p className="text-sm text-void-400">Refusal Rate</p>
            <p className="text-xl font-bold text-void-100">
              {((summary?.refusal_rate || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Status Distribution */}
        <Card>
          <CardHeader title="Request Status Distribution" subtitle="Breakdown of request outcomes" />
          {statusData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {statusData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#243b53', 
                      border: '1px solid #334e68',
                      borderRadius: '8px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-void-500">
              No data available
            </div>
          )}
        </Card>

        {/* Latency Distribution */}
        <Card>
          <CardHeader title="Latency Distribution" subtitle="Response time percentiles" />
          {latencyData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334e68" />
                  <XAxis dataKey="name" stroke="#627d98" />
                  <YAxis stroke="#627d98" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#243b53', 
                      border: '1px solid #334e68',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toFixed(0)}ms`, 'Latency']}
                  />
                  <Bar dataKey="value" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-void-500">
              No data available
            </div>
          )}
        </Card>

        {/* Token Usage */}
        <Card>
          <CardHeader title="Average Token Usage" subtitle="Input vs output tokens" />
          {tokenData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tokenData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334e68" />
                  <XAxis type="number" stroke="#627d98" />
                  <YAxis dataKey="name" type="category" stroke="#627d98" width={100} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#243b53', 
                      border: '1px solid #334e68',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [value.toFixed(0), 'Tokens']}
                  />
                  <Bar dataKey="avg" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-void-500">
              No data available
            </div>
          )}
        </Card>

        {/* Models Used */}
        <Card>
          <CardHeader title="Models & Providers" subtitle="Active configurations" />
          <div className="space-y-4">
            <div>
              <p className="text-sm text-void-400 mb-2">Models</p>
              <div className="flex flex-wrap gap-2">
                {summary?.models?.length ? summary.models.map((model) => (
                  <span key={model} className="px-3 py-1 bg-void-700 rounded-full text-sm text-void-200 font-mono">
                    {model}
                  </span>
                )) : (
                  <span className="text-void-500">No models</span>
                )}
              </div>
            </div>
            <div>
              <p className="text-sm text-void-400 mb-2">Providers</p>
              <div className="flex flex-wrap gap-2">
                {summary?.providers?.length ? summary.providers.map((provider) => (
                  <span key={provider} className="px-3 py-1 bg-void-700 rounded-full text-sm text-void-200">
                    {provider}
                  </span>
                )) : (
                  <span className="text-void-500">No providers</span>
                )}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
