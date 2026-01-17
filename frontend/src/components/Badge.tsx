import clsx from 'clsx'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info'

interface BadgeProps {
  children: React.ReactNode
  variant?: BadgeVariant
  size?: 'sm' | 'md'
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-void-700 text-void-200',
  success: 'bg-success-500/20 text-success-400 border border-success-500/30',
  warning: 'bg-warning-500/20 text-warning-400 border border-warning-500/30',
  danger: 'bg-accent-500/20 text-accent-400 border border-accent-500/30',
  info: 'bg-blue-500/20 text-blue-400 border border-blue-500/30',
}

export default function Badge({ children, variant = 'default', size = 'sm' }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center font-medium rounded-full',
        variantStyles[variant],
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
      )}
    >
      {children}
    </span>
  )
}
