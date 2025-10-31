# CortexOS Integration and Deployment Guide

## 🚀 Complete System Integration

This guide provides step-by-step instructions for integrating all CortexOS components and deploying the complete system for production use.

## 📋 Table of Contents
1. [Master Controller Setup](#master-controller-setup)
2. [Component Integration](#component-integration)
3. [Dashboard Integration](#dashboard-integration)
4. [Production Deployment](#production-deployment)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Scaling and Optimization](#scaling-and-optimization)
7. [Troubleshooting](#troubleshooting)

## 🎯 Master Controller Setup

### Create the Master Controller

First, create the main controller that orchestrates all CortexOS components:

```python
# /home/ubuntu/cortexos_rebuilt/cortexos_master_controller.py
#!/usr/bin/env python3
"""
CortexOS Master Controller
Orchestrates all 30 components across 6 phases
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add all phase directories to path
sys.path.extend([
    '/home/ubuntu/cortexos_rebuilt',
    '/home/ubuntu/cortexos_rebuilt/infrastructure',
    '/home/ubuntu/cortexos_rebuilt/phase1',
    '/home/ubuntu/cortexos_rebuilt/phase2',
    '/home/ubuntu/cortexos_rebuilt/phase3',
    '/home/ubuntu/cortexos_rebuilt/phase4',
    '/home/ubuntu/cortexos_rebuilt/phase5',
    '/home/ubuntu/cortexos_rebuilt/phase6'
])

class CortexOSMasterController:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.components = {}
        self.running = False
        self.startup_order = [
            # Infrastructure first
            'infrastructure',
            # Core engines
            'phase1', 
            # Resonance system
            'phase2',
            # Memory system
            'phase3',
            # Ingestion pipeline
            'phase4',
            # Monitoring system
            'phase5',
            # Modulation system
            'phase6'
        ]
        
        self.component_registry = {
            'infrastructure': [
                'path_manager', 'supervisor', 'cube_storage', 
                'contract_manager', 'sync_manager', 'neural_fabric'
            ],
            'phase1': ['neuroengine', 'context_engine', 'neural_gatekeeper'],
            'phase2': [
                'resonance_field', 'resonance_monitor', 
                'resonance_reinforcer', 'topk_sparse_resonance'
            ],
            'phase3': [
                'memory_inserter', 'memory_retriever', 
                'memory_consolidator', 'cognitive_bridge'
            ],
            'phase4': [
                'data_ingestion_engine', 'stream_processor', 
                'batch_processor', 'ingestion_validator'
            ],
            'phase5': [
                'system_monitor', 'performance_tracker', 
                'health_checker', 'alert_manager'
            ],
            'phase6': [
                'mood_modulator', 'cognitive_enhancer', 
                'neural_optimizer', 'adaptive_controller'
            ]
        }
        
        self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            "system": {
                "log_level": "INFO",
                "max_workers": 8,
                "memory_limit_gb": 16,
                "enable_gpu": False
            },
            "neural": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_iterations": 1000
            },
            "memory": {
                "cache_size_mb": 1024,
                "compression_enabled": True,
                "archival_threshold_days": 30
            },
            "monitoring": {
                "health_check_interval": 30,
                "performance_tracking": True,
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "error_rate": 5
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for section, settings in user_config.items():
                    if section in default_config:
                        default_config[section].update(settings)
                    else:
                        default_config[section] = settings
        
        return default_config
    
    def setup_logging(self):
        """Setup system logging"""
        log_level = getattr(logging, self.config['system']['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/tmp/cortexos_master.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CortexOS.Master')
    
    async def initialize_component(self, phase: str, component_name: str):
        """Initialize a single component"""
        try:
            self.logger.info(f"Initializing {phase}/{component_name}")
            
            # Dynamic import based on phase and component
            if phase == 'infrastructure':
                if component_name == 'path_manager':
                    from path_manager import PathManager
                    component = PathManager(self.config)
                elif component_name == 'supervisor':
                    from supervisor import Supervisor
                    component = Supervisor(self.config)
                elif component_name == 'cube_storage':
                    from cube_storage import CubeStorage
                    component = CubeStorage(self.config)
                elif component_name == 'contract_manager':
                    from contract_manager import ContractManager
                    component = ContractManager(self.config)
                elif component_name == 'sync_manager':
                    from sync_manager import SyncManager
                    component = SyncManager(self.config)
                elif component_name == 'neural_fabric':
                    from neural_fabric import NeuralFabric
                    component = NeuralFabric(self.config)
            
            elif phase == 'phase1':
                if component_name == 'neuroengine':
                    from neuroengine import NeuroEngine
                    component = NeuroEngine(self.config)
                elif component_name == 'context_engine':
                    from context_engine import ContextEngine
                    component = ContextEngine(self.config)
                elif component_name == 'neural_gatekeeper':
                    from neural_gatekeeper import NeuralGatekeeper
                    component = NeuralGatekeeper(self.config)
            
            # Add similar imports for other phases...
            
            # Initialize and start component
            await component.initialize()
            await component.start()
            
            self.components[f"{phase}.{component_name}"] = component
            self.logger.info(f"Successfully initialized {phase}/{component_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {phase}/{component_name}: {e}")
            raise
    
    async def start_system(self):
        """Start the complete CortexOS system"""
        self.logger.info("🧠 Starting CortexOS Master Controller")
        self.logger.info("=" * 60)
        
        self.running = True
        
        # Start components in dependency order
        for phase in self.startup_order:
            self.logger.info(f"Starting {phase.upper()} components...")
            
            # Start all components in this phase concurrently
            tasks = []
            for component_name in self.component_registry[phase]:
                task = asyncio.create_task(
                    self.initialize_component(phase, component_name)
                )
                tasks.append(task)
            
            # Wait for all components in this phase to start
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info(f"✅ {phase.upper()} phase complete")
            
            # Brief pause between phases
            await asyncio.sleep(2)
        
        self.logger.info("🎉 CortexOS system startup complete!")
        self.logger.info(f"📊 Total components running: {len(self.components)}")
        
        # Start monitoring loop
        await self.monitoring_loop()
    
    async def monitoring_loop(self):
        """Main monitoring and coordination loop"""
        self.logger.info("🔍 Starting monitoring loop")
        
        while self.running:
            try:
                # Health check all components
                healthy_count = 0
                total_count = len(self.components)
                
                for component_id, component in self.components.items():
                    try:
                        health = await component.health_check()
                        if health.get('status') == 'healthy':
                            healthy_count += 1
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {component_id}: {e}")
                
                health_percentage = (healthy_count / total_count) * 100 if total_count > 0 else 0
                
                self.logger.info(f"💚 System Health: {health_percentage:.1f}% ({healthy_count}/{total_count})")
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config['monitoring']['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def shutdown_system(self):
        """Gracefully shutdown the system"""
        self.logger.info("🛑 Shutting down CortexOS system...")
        self.running = False
        
        # Shutdown in reverse order
        for phase in reversed(self.startup_order):
            self.logger.info(f"Shutting down {phase.upper()} components...")
            
            for component_name in self.component_registry[phase]:
                component_id = f"{phase}.{component_name}"
                if component_id in self.components:
                    try:
                        await self.components[component_id].shutdown()
                        self.logger.info(f"✅ Shutdown {component_id}")
                    except Exception as e:
                        self.logger.error(f"Error shutting down {component_id}: {e}")
        
        self.logger.info("🏁 CortexOS shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown_system())

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CortexOS Master Controller')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Create and start controller
    controller = CortexOSMasterController(args.config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, controller.signal_handler)
    signal.signal(signal.SIGTERM, controller.signal_handler)
    
    try:
        await controller.start_system()
    except KeyboardInterrupt:
        await controller.shutdown_system()
    except Exception as e:
        controller.logger.error(f"Fatal error: {e}")
        await controller.shutdown_system()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Create System Configuration

Create a production configuration file:

```json
{
  "system": {
    "log_level": "INFO",
    "max_workers": 16,
    "memory_limit_gb": 32,
    "enable_gpu": true,
    "backup_enabled": true,
    "backup_interval_hours": 6
  },
  "neural": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_iterations": 5000,
    "optimization_enabled": true
  },
  "memory": {
    "cache_size_mb": 4096,
    "compression_enabled": true,
    "archival_threshold_days": 30,
    "consolidation_interval_hours": 4
  },
  "monitoring": {
    "health_check_interval": 15,
    "performance_tracking": true,
    "alert_thresholds": {
      "cpu_usage": 85,
      "memory_usage": 90,
      "error_rate": 3
    },
    "dashboard_enabled": true,
    "dashboard_port": 5001
  },
  "security": {
    "enable_encryption": true,
    "api_key_required": false,
    "rate_limiting": true,
    "max_requests_per_minute": 1000
  }
}
```

## 🔗 Component Integration

### Create Component Base Class

Create a base class that all components inherit from:

```python
# /home/ubuntu/cortexos_rebuilt/base_component.py
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseComponent(ABC):
    """Base class for all CortexOS components"""
    
    def __init__(self, config: Dict[str, Any], name: str):
        self.config = config
        self.name = name
        self.status = 'stopped'
        self.start_time = None
        self.logger = logging.getLogger(f'CortexOS.{name}')
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0,
            'throughput': 0.0,
            'latency': 0.0,
            'error_count': 0,
            'health_score': 1.0
        }
    
    @abstractmethod
    async def initialize(self):
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def start(self):
        """Start the component"""
        self.status = 'running'
        self.start_time = time.time()
        self.logger.info(f"{self.name} started successfully")
    
    @abstractmethod
    async def stop(self):
        """Stop the component"""
        self.status = 'stopped'
        self.logger.info(f"{self.name} stopped")
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data through the component"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'status': 'healthy' if self.status == 'running' else 'unhealthy',
            'uptime': uptime,
            'metrics': self.metrics.copy(),
            'last_check': time.time()
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        await self.stop()
```

### Update All Components

Update each component to inherit from BaseComponent and implement the required methods. Here's an example for the NeuroEngine:

```python
# Update /home/ubuntu/cortexos_rebuilt/phase1/neuroengine.py
from base_component import BaseComponent

class NeuroEngine(BaseComponent):
    def __init__(self, config):
        super().__init__(config, 'NeuroEngine')
        # ... existing initialization code ...
    
    async def initialize(self):
        """Initialize the neural engine"""
        self.logger.info("Initializing NeuroEngine...")
        # ... initialization logic ...
    
    async def start(self):
        """Start the neural engine"""
        await super().start()
        # ... start logic ...
    
    async def stop(self):
        """Stop the neural engine"""
        await super().stop()
        # ... stop logic ...
    
    async def process(self, data):
        """Process neural data"""
        # ... processing logic ...
        return processed_data
```

## 🌐 Dashboard Integration

### Connect Dashboard to Real Components

Update the dashboard to connect to actual components instead of mock data:

```python
# Update /home/ubuntu/cortexos_rebuilt/cortexos_dashboard/src/routes/cortexos_api.py

import sys
sys.path.append('/home/ubuntu/cortexos_rebuilt')

from cortexos_master_controller import CortexOSMasterController

# Global controller instance
controller = None

def get_controller():
    global controller
    if controller is None:
        controller = CortexOSMasterController()
    return controller

@cortexos_bp.route('/status')
def get_system_status():
    """Get real system status"""
    ctrl = get_controller()
    
    if not ctrl.running:
        return jsonify({
            'system_state': 'stopped',
            'overall_health': 0.0,
            'total_components': 30,
            'running_components': 0,
            'stopped_components': 30,
            'error_components': 0,
            'uptime_seconds': 0,
            'last_update': datetime.now().isoformat()
        })
    
    # Get real component status
    total_components = len(ctrl.components)
    running_components = sum(1 for comp in ctrl.components.values() 
                           if comp.status == 'running')
    
    # Calculate overall health
    health_scores = []
    for comp in ctrl.components.values():
        try:
            health = asyncio.run(comp.health_check())
            health_scores.append(health['metrics']['health_score'])
        except:
            health_scores.append(0.0)
    
    overall_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
    
    return jsonify({
        'system_state': 'running' if running_components > 0 else 'stopped',
        'overall_health': round(overall_health, 3),
        'total_components': total_components,
        'running_components': running_components,
        'stopped_components': total_components - running_components,
        'error_components': 0,  # Calculate from actual errors
        'uptime_seconds': int(time.time() - ctrl.start_time) if hasattr(ctrl, 'start_time') else 0,
        'last_update': datetime.now().isoformat()
    })
```

## 🚀 Production Deployment

### Create Deployment Scripts

Create deployment scripts for different environments:

#### Development Deployment
```bash
#!/bin/bash
# /home/ubuntu/cortexos_rebuilt/deploy_dev.sh

echo "🧠 Deploying CortexOS - Development Environment"

# Set environment
export CORTEXOS_ENV=development
export CORTEXOS_LOG_LEVEL=DEBUG

# Start master controller
cd /home/ubuntu/cortexos_rebuilt
python3 cortexos_master_controller.py --config cortexos_config.json &

# Start dashboard
cd cortexos_dashboard
source venv/bin/activate
python src/main.py &

echo "✅ CortexOS Development deployment complete"
echo "📊 Dashboard: http://localhost:5001"
echo "🔧 API: http://localhost:5001/api"
```

#### Production Deployment
```bash
#!/bin/bash
# /home/ubuntu/cortexos_rebuilt/deploy_prod.sh

echo "🧠 Deploying CortexOS - Production Environment"

# Set environment
export CORTEXOS_ENV=production
export CORTEXOS_LOG_LEVEL=WARNING

# Create production directories
mkdir -p /var/log/cortexos
mkdir -p /var/lib/cortexos/data
mkdir -p /var/lib/cortexos/backups

# Set permissions
chmod 755 /var/log/cortexos
chmod 755 /var/lib/cortexos

# Start with systemd service
sudo systemctl start cortexos
sudo systemctl enable cortexos

# Start dashboard service
sudo systemctl start cortexos-dashboard
sudo systemctl enable cortexos-dashboard

echo "✅ CortexOS Production deployment complete"
echo "📊 Dashboard: http://your-server:5001"
```

### Create Systemd Services

Create systemd service files for production:

```ini
# /etc/systemd/system/cortexos.service
[Unit]
Description=CortexOS Master Controller
After=network.target

[Service]
Type=simple
User=cortexos
Group=cortexos
WorkingDirectory=/home/ubuntu/cortexos_rebuilt
ExecStart=/usr/bin/python3 cortexos_master_controller.py --config /etc/cortexos/production.json
Restart=always
RestartSec=10
Environment=CORTEXOS_ENV=production
Environment=CORTEXOS_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/cortexos-dashboard.service
[Unit]
Description=CortexOS Web Dashboard
After=network.target cortexos.service

[Service]
Type=simple
User=cortexos
Group=cortexos
WorkingDirectory=/home/ubuntu/cortexos_rebuilt/cortexos_dashboard
ExecStart=/home/ubuntu/cortexos_rebuilt/cortexos_dashboard/venv/bin/python src/main.py
Restart=always
RestartSec=10
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

Create Docker containers for easy deployment:

```dockerfile
# /home/ubuntu/cortexos_rebuilt/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create cortexos user
RUN useradd -m -s /bin/bash cortexos

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R cortexos:cortexos /app

# Switch to cortexos user
USER cortexos

# Expose ports
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/status || exit 1

# Start command
CMD ["python", "cortexos_master_controller.py", "--config", "cortexos_production.json"]
```

```yaml
# /home/ubuntu/cortexos_rebuilt/docker-compose.yml
version: '3.8'

services:
  cortexos:
    build: .
    container_name: cortexos-system
    restart: unless-stopped
    ports:
      - "5001:5001"
    volumes:
      - cortexos_data:/app/data
      - cortexos_logs:/app/logs
      - ./cortexos_production.json:/app/cortexos_production.json:ro
    environment:
      - CORTEXOS_ENV=production
      - CORTEXOS_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  cortexos-dashboard:
    build: 
      context: ./cortexos_dashboard
    container_name: cortexos-dashboard
    restart: unless-stopped
    ports:
      - "8080:5001"
    depends_on:
      - cortexos
    environment:
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  cortexos_data:
  cortexos_logs:
```

## 📊 Monitoring and Maintenance

### Create Monitoring Scripts

```bash
#!/bin/bash
# /home/ubuntu/cortexos_rebuilt/monitor.sh

echo "🔍 CortexOS System Monitor"
echo "=========================="

# Check system status
echo "📊 System Status:"
curl -s http://localhost:5001/api/status | jq '.'

echo ""
echo "🔧 Component Status:"
curl -s http://localhost:5001/api/components | jq '.[] | keys'

echo ""
echo "💾 Memory Usage:"
curl -s http://localhost:5001/api/memory/stats | jq '.'

echo ""
echo "⚡ Neural Performance:"
curl -s http://localhost:5001/api/neural/performance | jq '.'

echo ""
echo "🚨 Recent Alerts:"
curl -s http://localhost:5001/api/alerts | jq '.[] | select(.status == "active")'
```

### Create Backup Scripts

```bash
#!/bin/bash
# /home/ubuntu/cortexos_rebuilt/backup.sh

BACKUP_DIR="/var/lib/cortexos/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="cortexos_backup_${TIMESTAMP}.tar.gz"

echo "💾 Creating CortexOS backup: ${BACKUP_FILE}"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Backup system data
tar -czf ${BACKUP_DIR}/${BACKUP_FILE} \
    /home/ubuntu/cortexos_rebuilt \
    /var/log/cortexos \
    /var/lib/cortexos/data \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv'

echo "✅ Backup complete: ${BACKUP_DIR}/${BACKUP_FILE}"

# Clean old backups (keep last 7 days)
find ${BACKUP_DIR} -name "cortexos_backup_*.tar.gz" -mtime +7 -delete

echo "🧹 Old backups cleaned"
```

## 📈 Scaling and Optimization

### Horizontal Scaling

For high-load environments, implement horizontal scaling:

```python
# /home/ubuntu/cortexos_rebuilt/cortexos_cluster_manager.py
class CortexOSClusterManager:
    def __init__(self, config):
        self.config = config
        self.nodes = []
        self.load_balancer = LoadBalancer()
    
    async def add_node(self, node_address: str):
        """Add a new node to the cluster"""
        node = CortexOSNode(node_address)
        await node.connect()
        self.nodes.append(node)
        self.load_balancer.register_node(node)
    
    async def distribute_load(self, request):
        """Distribute requests across nodes"""
        node = self.load_balancer.select_node()
        return await node.process_request(request)
    
    async def health_check_cluster(self):
        """Check health of all nodes"""
        for node in self.nodes:
            health = await node.health_check()
            if health['status'] != 'healthy':
                self.load_balancer.mark_unhealthy(node)
```

### Performance Optimization

```python
# /home/ubuntu/cortexos_rebuilt/performance_optimizer.py
class PerformanceOptimizer:
    def __init__(self, controller):
        self.controller = controller
        self.metrics_history = []
    
    async def optimize_system(self):
        """Automatically optimize system performance"""
        current_metrics = await self.collect_metrics()
        
        # CPU optimization
        if current_metrics['cpu_usage'] > 80:
            await self.scale_workers()
        
        # Memory optimization
        if current_metrics['memory_usage'] > 85:
            await self.trigger_memory_cleanup()
        
        # Neural optimization
        if current_metrics['neural_efficiency'] < 0.8:
            await self.optimize_neural_networks()
    
    async def scale_workers(self):
        """Scale worker processes based on load"""
        current_workers = self.controller.config['system']['max_workers']
        new_workers = min(current_workers * 2, 32)
        
        self.controller.config['system']['max_workers'] = new_workers
        await self.controller.restart_worker_pool()
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Component Startup Failures
```bash
# Check component logs
tail -f /tmp/cortexos_*.log

# Restart specific component
curl -X POST http://localhost:5001/api/components/phase1/neuroengine/restart

# Check dependencies
python3 -c "import sys; sys.path.append('/home/ubuntu/cortexos_rebuilt'); from phase1.neuroengine import NeuroEngine; print('Import successful')"
```

#### 2. Memory Issues
```bash
# Check memory usage
curl http://localhost:5001/api/memory/stats

# Trigger memory consolidation
curl -X POST http://localhost:5001/api/memory/consolidate

# Clear cache
curl -X POST http://localhost:5001/api/memory/clear-cache
```

#### 3. Performance Issues
```bash
# Check system performance
curl http://localhost:5001/api/analytics/performance

# Request optimization
curl -X POST http://localhost:5001/api/neural/optimization/request \
  -H "Content-Type: application/json" \
  -d '{"method": "evolutionary", "targets": ["throughput", "latency"]}'
```

### Diagnostic Tools

```bash
#!/bin/bash
# /home/ubuntu/cortexos_rebuilt/diagnose.sh

echo "🔍 CortexOS System Diagnostics"
echo "=============================="

# System resources
echo "💻 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
echo "Memory Usage: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"

# Process status
echo ""
echo "🔄 Process Status:"
ps aux | grep -E "(cortexos|python)" | grep -v grep

# Network connectivity
echo ""
echo "🌐 Network Status:"
netstat -tulpn | grep -E "(5001|8080)"

# Log analysis
echo ""
echo "📋 Recent Errors:"
tail -n 50 /tmp/cortexos_*.log | grep -i error | tail -10

# Component health
echo ""
echo "🏥 Component Health:"
curl -s http://localhost:5001/api/components | jq '.[] | to_entries[] | select(.value.status != "running") | .key'
```

## 🎯 Quick Start Commands

### Development Environment
```bash
# Start development environment
cd /home/ubuntu/cortexos_rebuilt
./deploy_dev.sh

# Monitor system
./monitor.sh

# View logs
tail -f /tmp/cortexos_*.log
```

### Production Environment
```bash
# Deploy to production
./deploy_prod.sh

# Check status
systemctl status cortexos cortexos-dashboard

# Monitor performance
./monitor.sh

# Create backup
./backup.sh
```

### Docker Environment
```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale cortexos=3
```

## 📚 Additional Resources

- **API Documentation**: Available at `http://localhost:5001/docs`
- **Component Documentation**: See individual component files
- **Configuration Reference**: See `cortexos_config.json` examples
- **Monitoring Dashboards**: Access at `http://localhost:5001`

---

**🎉 Congratulations! You now have a complete, production-ready CortexOS system with full integration, monitoring, and deployment capabilities.**

