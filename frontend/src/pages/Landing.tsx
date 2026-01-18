import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowRight, 
  Sparkles,
  Shield,
  BarChart3,
  Brain,
  GitBranch
} from 'lucide-react';

const Landing = () => {
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const nodes: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
    }> = [];

    // Create nodes
    for (let i = 0; i < 50; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1,
      });
    }

    const animate = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw nodes
      nodes.forEach((node, i) => {
        node.x += node.vx;
        node.y += node.vy;

        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.fill();

        // Draw connections
        nodes.slice(i + 1).forEach((otherNode) => {
          const dx = otherNode.x - node.x;
          const dy = otherNode.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(otherNode.x, otherNode.y);
            ctx.strokeStyle = `rgba(239, 68, 68, ${0.2 * (1 - distance / 150)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const features = [
    {
      icon: GitBranch,
      title: 'Intelligent Replay Engine',
      description: 'Replay production prompts across candidate models with deterministic evaluation and real-time cost tracking',
    },
    {
      icon: Brain,
      title: 'AI Judge Evaluation',
      description: 'Multi-judge system (Comparison, Quality, Helpfulness) evaluates responses with confidence scoring and disagreement detection',
    },
    {
      icon: Sparkles,
      title: 'Smart Model Recommendations',
      description: 'Confidence-gated recommendations with full trade-off analysis - outputs "NO RECOMMENDATION" when uncertain',
    },
    {
      icon: BarChart3,
      title: 'Safe Analytics Engine',
      description: 'Statistical analysis with distributions, percentiles, correlations, clustering - no arbitrary code execution',
    },
    {
      icon: Shield,
      title: 'Production Log Ingestion',
      description: 'Sync logs from Portkey with immutable storage, version tracking, and optional privacy mode',
    },
  ];

  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      {/* Animated Background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 opacity-30"
      />

      {/* Gradient Overlays */}
      <div className="absolute inset-0 bg-gradient-to-br from-red-900/20 via-transparent to-orange-900/20" />
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-red-500/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-orange-500/10 rounded-full blur-3xl animate-pulse delay-1000" />

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="container mx-auto px-6 py-8">
          <nav className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-8 h-8 text-red-400" />
              <div className="flex flex-col">
                <span className="text-2xl font-bold bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent">
                  Clot
                </span>
                <span className="text-xs text-gray-400 -mt-1">
                  Stop Bleeding Your Money
                </span>
              </div>
            </div>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="container mx-auto px-6 pt-20 pb-32">
          <div className={`max-w-4xl mx-auto text-center transition-all duration-1000 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
            <h1 className="text-6xl md:text-7xl font-bold mb-6 leading-tight">
              Optimize Your Costs
              <br />
              <span className="bg-gradient-to-r from-red-400 via-rose-400 to-pink-400 bg-clip-text text-transparent">
                With Intelligence
              </span>
            </h1>

            <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              Production-grade platform for analyzing AI agent performance and receiving 
              explainable model recommendations based on real-world data.
            </p>

            <button
              onClick={() => navigate('/dashboard')}
              className="group relative inline-flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-red-600 to-rose-600 rounded-full text-lg font-semibold hover:shadow-2xl hover:shadow-red-500/50 transition-all duration-300 hover:scale-105"
            >
              <span>Explore Platform</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </section>

        {/* Features Grid */}
        <section className="container mx-auto px-6 py-20">
          <div className={`text-center mb-16 transition-all duration-1000 delay-300 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent">
                Features
              </span>
            </h2>
            <p className="text-gray-400 text-lg">
              Everything you need to optimize and monitor your AI agents
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
            {features.map((feature, index) => (
              <div
                key={index}
                className={`group relative p-8 bg-gradient-to-br from-gray-900/50 to-gray-800/30 backdrop-blur-sm border border-gray-800 rounded-2xl hover:border-red-500/50 transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-red-500/20 ${
                  isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
                }`}
                style={{ transitionDelay: `${400 + index * 100}ms` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-rose-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
                <div className="relative">
                  <div className="w-14 h-14 bg-gradient-to-br from-red-500/20 to-rose-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <feature.icon className="w-7 h-7 text-red-400" />
                  </div>
                  
                  <h3 className="text-xl font-semibold mb-3 text-white">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Architecture Diagram */}
        <section className="container mx-auto px-6 py-20">
          <div className={`text-center mb-16 transition-all duration-1000 delay-700 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent">
                Architecture
              </span>
            </h2>
            <p className="text-gray-400 text-lg">
              How components work together to optimize your AI agents
            </p>
          </div>

          <div className="max-w-6xl mx-auto">
            <ArchitectureDiagram />
          </div>
        </section>

        {/* How It Works - Simulation */}
        <section className="container mx-auto px-6 py-20">
          <div className={`text-center mb-16 transition-all duration-1000 delay-800 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              How It{' '}
              <span className="bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent">
                Works
              </span>
            </h2>
            <p className="text-gray-400 text-lg">
              Automated evaluation pipeline from logs to recommendations
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            <WorkflowSimulation />
          </div>
        </section>

        {/* CTA Section */}
        <section className="container mx-auto px-6 py-20">
          <div className={`max-w-4xl mx-auto text-center transition-all duration-1000 delay-900 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
          }`}>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to{' '}
              <span className="bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent">
                Transform
              </span>
              {' '}Your AI Operations?
            </h2>
            <p className="text-xl text-gray-400 mb-10">
              Start optimizing your AI agents with intelligent insights today
            </p>
            <button
              onClick={() => navigate('/dashboard')}
              className="group relative inline-flex items-center space-x-2 px-10 py-5 bg-white text-black rounded-full text-lg font-semibold hover:shadow-2xl hover:shadow-white/30 transition-all duration-300 hover:scale-105"
            >
              <span>Get Started</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer className="container mx-auto px-6 py-12 border-t border-gray-800">
          <div className="flex items-center justify-center space-x-2 text-gray-500">
            <Sparkles className="w-5 h-5" />
            <span>© 2026 Clot. Stop Bleeding Your Money.</span>
          </div>
        </footer>
      </div>
    </div>
  );
};

const ArchitectureDiagram = () => {
  return (
    <div className="relative p-8 bg-gradient-to-br from-gray-900/50 to-gray-800/30 backdrop-blur-sm border border-gray-800 rounded-2xl">
      <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-rose-500/5 rounded-2xl" />
      
      <div className="relative">
        <div className="bg-gray-900/80 rounded-xl p-6 border border-gray-700 overflow-hidden">
          <img 
            src="/architecture-diagram.png" 
            alt="Platform Architecture Diagram" 
            className="w-full h-auto rounded-lg"
          />
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-sm text-gray-400">
            System architecture showing data flow between components
          </p>
        </div>
      </div>
    </div>
  );
};

const WorkflowSimulation = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: 'Ingest Production Logs',
      description: 'Sync logs from Portkey with immutable storage',
      icon: '📥',
      color: 'from-red-500 to-rose-500',
    },
    {
      title: 'Analyze Workload',
      description: 'Statistical analysis: distributions, correlations, clustering',
      icon: '📊',
      color: 'from-rose-500 to-pink-500',
    },
    {
      title: 'Select Candidate Models',
      description: 'AI-powered selection with pruning rules + LiveBench benchmarks per user preference',
      icon: '🎯',
      color: 'from-pink-500 to-red-500',
    },
    {
      title: 'Replay Prompts',
      description: 'Deterministic replay across models via Portkey',
      icon: '🔄',
      color: 'from-red-500 to-orange-500',
    },
    {
      title: 'AI Judge Evaluation',
      description: 'Multi-judge scoring: Comparison, Quality, Helpfulness',
      icon: '⚖️',
      color: 'from-orange-500 to-rose-500',
    },
    {
      title: 'Generate Recommendation',
      description: 'Confidence-gated with full trade-off analysis',
      icon: '✨',
      color: 'from-rose-500 to-red-500',
    },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 3000);

    return () => clearInterval(interval);
  }, [steps.length]);

  return (
    <div className="relative">
      {/* Progress Line */}
      <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-red-500/20 via-rose-500/20 to-pink-500/20" />
      
      {/* Active Progress */}
      <div 
        className="absolute left-8 top-0 w-0.5 bg-gradient-to-b from-red-500 via-rose-500 to-pink-500 transition-all duration-1000 ease-out"
        style={{ height: `${((activeStep + 1) / steps.length) * 100}%` }}
      />

      {/* Steps */}
      <div className="space-y-8">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`relative flex items-start gap-6 transition-all duration-500 ${
              index === activeStep ? 'scale-105' : 'scale-100 opacity-60'
            }`}
          >
            {/* Icon */}
            <div
              className={`relative z-10 w-16 h-16 rounded-2xl flex items-center justify-center text-2xl transition-all duration-500 ${
                index <= activeStep
                  ? `bg-gradient-to-br ${step.color} shadow-lg shadow-red-500/50`
                  : 'bg-gray-800 border border-gray-700'
              }`}
            >
              {step.icon}
            </div>

            {/* Content */}
            <div className="flex-1 pt-2">
              <h3
                className={`text-xl font-semibold mb-2 transition-colors duration-500 ${
                  index === activeStep ? 'text-white' : 'text-gray-400'
                }`}
              >
                {step.title}
              </h3>
              <p
                className={`text-sm transition-colors duration-500 ${
                  index === activeStep ? 'text-gray-300' : 'text-gray-500'
                }`}
              >
                {step.description}
              </p>

              {/* Animated Progress Bar */}
              {index === activeStep && (
                <div className="mt-3 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${step.color} animate-progress`}
                    style={{ animation: 'progress 3s linear' }}
                  />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes progress {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
};

export default Landing;
