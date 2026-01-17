import { ReactNode } from 'react'
import clsx from 'clsx'

interface CardProps {
  children: ReactNode
  className?: string
  hover?: boolean
}

export default function Card({ children, className, hover = false }: CardProps) {
  return (
    <div
      className={clsx(
        'glass-panel p-6',
        hover && 'card-hover cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps {
  title: string
  subtitle?: string
  action?: ReactNode
}

export function CardHeader({ title, subtitle, action }: CardHeaderProps) {
  return (
    <div className="flex items-start justify-between mb-4">
      <div>
        <h3 className="font-display font-semibold text-lg text-void-50">{title}</h3>
        {subtitle && <p className="text-sm text-void-400 mt-1">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string | number
  change?: number
  changeLabel?: string
  icon?: ReactNode
}

export function MetricCard({ label, value, change, changeLabel, icon }: MetricCardProps) {
  const isPositive = change !== undefined && change >= 0
  
  return (
    <Card>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-void-400 mb-1">{label}</p>
          <p className="text-2xl font-display font-bold text-void-50">{value}</p>
          {change !== undefined && (
            <p className={clsx(
              'text-sm mt-2 flex items-center gap-1',
              isPositive ? 'text-success-500' : 'text-accent-500'
            )}>
              <span>{isPositive ? '↑' : '↓'}</span>
              <span>{Math.abs(change).toFixed(1)}%</span>
              {changeLabel && <span className="text-void-500">{changeLabel}</span>}
            </p>
          )}
        </div>
        {icon && (
          <div className="w-12 h-12 rounded-xl bg-void-800 flex items-center justify-center text-void-400">
            {icon}
          </div>
        )}
      </div>
    </Card>
  )
}
