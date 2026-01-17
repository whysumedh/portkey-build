import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Folder, 
  Sparkles,
  Settings,
  ChevronRight
} from 'lucide-react'
import clsx from 'clsx'

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/projects', label: 'Projects', icon: Folder },
]

export default function Layout() {
  const location = useLocation()
  
  // Extract breadcrumbs from path
  const pathParts = location.pathname.split('/').filter(Boolean)
  
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-void-900/80 border-r border-void-700/50 flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-void-700/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-display font-semibold text-void-50">Agent Optimizer</h1>
              <p className="text-xs text-void-400">Model Selection Platform</p>
            </div>
          </div>
        </div>
        
        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-1">
            {navItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200',
                      isActive
                        ? 'bg-accent-500/20 text-accent-400 border border-accent-500/30'
                        : 'text-void-300 hover:bg-void-800 hover:text-void-100'
                    )
                  }
                >
                  <item.icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        
        {/* Footer */}
        <div className="p-4 border-t border-void-700/50">
          <button className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-void-400 hover:bg-void-800 hover:text-void-200 transition-colors">
            <Settings className="w-5 h-5" />
            <span>Settings</span>
          </button>
        </div>
      </aside>
      
      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Breadcrumb */}
        <header className="h-16 border-b border-void-700/50 flex items-center px-6 bg-void-900/40">
          <nav className="flex items-center gap-2 text-sm">
            {pathParts.map((part, index) => (
              <div key={index} className="flex items-center gap-2">
                {index > 0 && <ChevronRight className="w-4 h-4 text-void-500" />}
                <span className={clsx(
                  index === pathParts.length - 1 ? 'text-void-100' : 'text-void-400'
                )}>
                  {part.charAt(0).toUpperCase() + part.slice(1)}
                </span>
              </div>
            ))}
          </nav>
        </header>
        
        {/* Page content */}
        <div className="flex-1 overflow-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
