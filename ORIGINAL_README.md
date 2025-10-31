# CortexOS - Complete Setup and Usage Guide

## 🧠 Welcome to CortexOS
CortexOS is a comprehensive cognitive architecture system with 30 interconnected components across 6 phases, designed for advanced neural processing, memory management, and adaptive control.

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Component Overview](#component-overview)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [Web UI Dashboard](#web-ui-dashboard)
8. [API Usage](#api-usage)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **CPU**: 4 cores minimum

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11+
- **RAM**: 32GB+ (for optimal performance)
- **Storage**: 10GB+ SSD
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **GPU**: Optional but recommended for neural optimization

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Navigate to your CortexOS directory
cd /home/ubuntu/cortexos_rebuilt

# Install required dependencies
pip install asyncio numpy pandas matplotlib seaborn plotly flask fastapi uvicorn websockets aiofiles
```

### 2. Test Individual Components
```bash
# Test infrastructure components
python3 infrastructure/cube_storage.py
python3 infrastructure/neural_fabric.py

# Test core engines
python3 phase1/neuroengine.py
python3 phase1/context_engine.py

# Test memory system
python3 phase3/memory_inserter.py
python3 phase3/cognitive_bridge.py
```

### 3. Start the Complete System
```bash
# Run the master controller
python3 cortexos_master_controller.py
```

## 🏗️ Component Overview

### Infrastructure Layer (6 components)
- **Path Manager** (`0_path_manager.py`) - File system and path management
- **Supervisor** (`1_supervisor.py`) - Process supervision and coordination
- **Cube Storage** (`infrastructure/cube_storage.py`) - Data storage management
- **Contract Manager** (`infrastructure/contract_manager.py`) - Service contracts
- **Sync Manager** (`infrastructure/sync_manager.py`) - Data synchronization
- **Neural Fabric** (`infrastructure/neural_fabric.py`) - Neural network foundation

### Phase 1: Core Engines (3 components)
- **Neuroengine** (`phase1/neuroengine.py`) - Primary neural processing
- **Context Engine** (`phase1/context_engine.py`) - Context understanding
- **Neural Gatekeeper** (`phase1/neural_gatekeeper.py`) - Access control

### Phase 2: Resonance System (4 components)
- **Resonance Field** (`phase2/resonance_field.py`) - Pattern resonance detection
- **Resonance Monitor** (`phase2/resonance_monitor.py`) - Real-time monitoring
- **Resonance Reinforcer** (`phase2/resonance_reinforcer.py`) - Pattern reinforcement
- **TopK Sparse Resonance** (`phase2/topk_sparse_resonance.py`) - Sparse processing

### Phase 3: Memory System (4 components)
- **Memory Inserter** (`phase3/memory_inserter.py`) - Memory storage
- **Memory Retriever** (`phase3/memory_retriever.py`) - Memory retrieval
- **Memory Consolidator** (`phase3/memory_consolidator.py`) - Memory optimization
- **Cognitive Bridge** (`phase3/cognitive_bridge.py`) - Neural-memory interface

### Phase 4: Ingestion Pipeline (4 components)
- **Data Ingestion Engine** (`phase4/data_ingestion_engine.py`) - Data input processing
- **Stream Processor** (`phase4/stream_processor.py`) - Real-time streaming
- **Batch Processor** (`phase4/batch_processor.py`) - Batch processing
- **Ingestion Validator** (`phase4/ingestion_validator.py`) - Data validation

### Phase 5: Monitoring System (4 components)
- **System Monitor** (`phase5/system_monitor.py`) - System health monitoring
- **Performance Tracker** (`phase5/performance_tracker.py`) - Performance analysis
- **Health Checker** (`phase5/health_checker.py`) - Health diagnostics
- **Alert Manager** (`phase5/alert_manager.py`) - Alert management

### Phase 6: Modulation System (4 components)
- **Mood Modulator** (`phase6/mood_modulator.py`) - Emotional state management
- **Cognitive Enhancer** (`phase6/cognitive_enhancer.py`) - Performance enhancement
- **Neural Optimizer** (`phase6/neural_optimizer.py`) - Neural network optimization
- **Adaptive Controller** (`phase6/adaptive_controller.py`) - Master system control

## 📦 Installation

### Step 1: Verify Python Installation
```bash
python3 --version  # Should be 3.8+
pip3 --version
```

### Step 2: Install Dependencies
```bash
# Core dependencies
pip3 install asyncio numpy pandas matplotlib seaborn plotly

# Web framework dependencies
pip3 install flask fastapi uvicorn websockets aiofiles

# Optional: GPU acceleration (if you have compatible GPU)
pip3 install torch torchvision  # For PyTorch
# OR
pip3 install tensorflow  # For TensorFlow
```

### Step 3: Verify Installation
```bash
# Test basic imports
python3 -c "import asyncio, numpy, pandas, matplotlib; print('Dependencies OK')"
```

## ⚙️ Configuration

### Basic Configuration
Create a configuration file `cortexos_config.json`:

```json
{
  "system": {
    "log_level": "INFO",
    "max_workers": 8,
    "memory_limit_gb": 16,
    "enable_gpu": false
  },
  "neural": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_iterations": 1000
  },
  "memory": {
    "cache_size_mb": 1024,
    "compression_enabled": true,
    "archival_threshold_days": 30
  },
  "monitoring": {
    "health_check_interval": 30,
    "performance_tracking": true,
    "alert_thresholds": {
      "cpu_usage": 80,
      "memory_usage": 85,
      "error_rate": 5
    }
  },
  "web_ui": {
    "host": "0.0.0.0",
    "port": 8080,
    "enable_auth": false
  }
}
```

### Advanced Configuration
For production environments, create `cortexos_production.json`:

```json
{
  "system": {
    "log_level": "WARNING",
    "max_workers": 16,
    "memory_limit_gb": 64,
    "enable_gpu": true,
    "backup_enabled": true,
    "backup_interval_hours": 6
  },
  "security": {
    "enable_encryption": true,
    "api_key_required": true,
    "rate_limiting": true,
    "max_requests_per_minute": 1000
  },
  "clustering": {
    "enable_distributed": true,
    "cluster_nodes": ["node1:8080", "node2:8080"],
    "load_balancing": "round_robin"
  }
}
```

## 🏃 Running the System

### Method 1: Individual Component Testing
```bash
# Test each component individually
cd /home/ubuntu/cortexos_rebuilt

# Infrastructure
python3 infrastructure/cube_storage.py
python3 infrastructure/neural_fabric.py

# Core engines
python3 phase1/neuroengine.py
python3 phase1/context_engine.py
python3 phase1/neural_gatekeeper.py

# Memory system
python3 phase3/memory_inserter.py
python3 phase3/memory_retriever.py
python3 phase3/cognitive_bridge.py

# Monitoring
python3 phase5/system_monitor.py
python3 phase5/health_checker.py

# Modulation
python3 phase6/mood_modulator.py
python3 phase6/neural_optimizer.py
python3 phase6/adaptive_controller.py
```

### Method 2: Integrated System (Coming Next)
```bash
# Start the master controller
python3 cortexos_master_controller.py

# Or with configuration
python3 cortexos_master_controller.py --config cortexos_config.json
```

### Method 3: Web UI Dashboard (Coming Next)
```bash
# Start the web interface
python3 cortexos_web_dashboard.py

# Access at: http://localhost:8080
```

## 🌐 Web UI Dashboard

The CortexOS Web Dashboard provides:

### Main Dashboard
- **System Overview**: Real-time status of all 30 components
- **Performance Metrics**: CPU, memory, throughput, latency
- **Health Monitoring**: Component health scores and alerts
- **Neural Activity**: Live neural processing visualization

### Component Management
- **Start/Stop Components**: Individual component control
- **Configuration**: Real-time parameter adjustment
- **Logs**: Component-specific log viewing
- **Dependencies**: Visual dependency graph

### Memory Management
- **Memory Usage**: Real-time memory consumption
- **Cache Status**: Cache hit rates and efficiency
- **Consolidation**: Manual memory consolidation triggers
- **Archival**: Historical data management

### Neural Optimization
- **Optimization Requests**: Submit optimization jobs
- **Performance Tracking**: Neural network performance
- **Learning Progress**: Training and adaptation metrics
- **Model Management**: Save/load neural configurations

### System Analytics
- **Performance Trends**: Historical performance data
- **Error Analysis**: Error patterns and resolution
- **Resource Utilization**: System resource usage
- **Adaptation History**: System adaptation timeline

## 🔌 API Usage

### REST API Endpoints

#### System Status
```bash
# Get overall system status
curl http://localhost:8080/api/status

# Get component status
curl http://localhost:8080/api/components/status

# Get specific component
curl http://localhost:8080/api/components/neuroengine/status
```

#### Component Control
```bash
# Start component
curl -X POST http://localhost:8080/api/components/neuroengine/start

# Stop component
curl -X POST http://localhost:8080/api/components/neuroengine/stop

# Restart component
curl -X POST http://localhost:8080/api/components/neuroengine/restart
```

#### Memory Operations
```bash
# Insert memory
curl -X POST http://localhost:8080/api/memory/insert \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "context": "example"}'

# Retrieve memory
curl http://localhost:8080/api/memory/search?query=test

# Consolidate memory
curl -X POST http://localhost:8080/api/memory/consolidate
```

#### Neural Operations
```bash
# Submit optimization request
curl -X POST http://localhost:8080/api/neural/optimize \
  -H "Content-Type: application/json" \
  -d '{"targets": ["throughput", "accuracy"], "method": "evolutionary"}'

# Get optimization status
curl http://localhost:8080/api/neural/optimization/status/12345

# Get neural performance
curl http://localhost:8080/api/neural/performance
```

### Python API Usage

```python
import asyncio
import requests
from cortexos_client import CortexOSClient

# Initialize client
client = CortexOSClient("http://localhost:8080")

# System operations
status = client.get_system_status()
print(f"System health: {status['health']}")

# Component operations
client.start_component("neuroengine")
client.stop_component("memory_inserter")

# Memory operations
memory_id = client.insert_memory("Important information", context="user_input")
results = client.search_memory("Important")

# Neural operations
optimization_id = client.request_optimization(
    targets=["throughput", "accuracy"],
    method="evolutionary"
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Install missing dependencies
pip3 install [missing_module]

# Error: Python version incompatible
# Solution: Upgrade Python to 3.8+
```

#### 2. Memory Issues
```bash
# Error: Out of memory
# Solution: Reduce batch size or increase system RAM
# Edit cortexos_config.json:
{
  "neural": {
    "batch_size": 16  // Reduce from 32
  },
  "memory": {
    "cache_size_mb": 512  // Reduce from 1024
  }
}
```

#### 3. Component Startup Failures
```bash
# Check component dependencies
python3 -c "
import sys
sys.path.append('/home/ubuntu/cortexos_rebuilt')
from phase1.neuroengine import NeuroEngine
print('Import successful')
"

# Check for port conflicts
netstat -tulpn | grep :8080
```

#### 4. Performance Issues
```bash
# Monitor system resources
htop
# Or
top

# Check component health
curl http://localhost:8080/api/components/status
```

### Debug Mode
```bash
# Run with debug logging
export CORTEXOS_LOG_LEVEL=DEBUG
python3 cortexos_master_controller.py

# Or modify config
{
  "system": {
    "log_level": "DEBUG"
  }
}
```

### Log Analysis
```bash
# View system logs
tail -f /tmp/cortexos_system.log

# View component logs
tail -f /tmp/cortexos_neuroengine.log
tail -f /tmp/cortexos_memory.log
```

## 🚀 Advanced Usage

### Custom Component Development
```python
# Create custom component
from cortexos.base import BaseComponent

class CustomProcessor(BaseComponent):
    def __init__(self, config):
        super().__init__(config)
        self.name = "custom_processor"
    
    async def start(self):
        await super().start()
        # Custom initialization
    
    async def process(self, data):
        # Custom processing logic
        return processed_data

# Register with system
controller.register_component("custom_processor", CustomProcessor)
```

### Distributed Deployment
```bash
# Node 1 (Master)
python3 cortexos_master_controller.py --mode master --port 8080

# Node 2 (Worker)
python3 cortexos_master_controller.py --mode worker --master-host node1 --port 8081

# Node 3 (Worker)
python3 cortexos_master_controller.py --mode worker --master-host node1 --port 8082
```

### Performance Tuning
```json
{
  "performance": {
    "neural_threads": 8,
    "memory_threads": 4,
    "io_threads": 2,
    "batch_processing": true,
    "async_operations": true,
    "cache_optimization": true
  }
}
```

### Monitoring Integration
```bash
# Prometheus metrics
curl http://localhost:8080/metrics

# Grafana dashboard
# Import cortexos_grafana_dashboard.json

# Custom alerts
curl -X POST http://localhost:8080/api/alerts/create \
  -d '{"condition": "cpu_usage > 90", "action": "scale_up"}'
```

## 📚 Next Steps

1. **Start with Individual Components**: Test each component to understand functionality
2. **Use the Web Dashboard**: Visual interface for system management
3. **Explore the API**: Integrate CortexOS with your applications
4. **Customize Configuration**: Tune for your specific use case
5. **Monitor Performance**: Use built-in monitoring tools
6. **Scale as Needed**: Add more nodes for distributed processing

## 🆘 Support

- **Documentation**: This guide and component-specific docs
- **Logs**: Check `/tmp/cortexos_*.log` files
- **API Reference**: Available at `http://localhost:8080/docs`
- **Component Tests**: Each component has built-in test functions

## 🔄 Updates and Maintenance

### Regular Maintenance
```bash
# Update dependencies
pip3 install --upgrade -r requirements.txt

# Clean up logs
find /tmp -name "cortexos_*.log" -mtime +7 -delete

# Backup configuration
cp cortexos_config.json cortexos_config.backup.$(date +%Y%m%d)
```

### System Health Checks
```bash
# Daily health check
python3 -c "
from cortexos_health_check import run_health_check
run_health_check()
"

# Performance benchmark
python3 cortexos_benchmark.py
```

---

**🎉 Congratulations! You now have a complete guide to using CortexOS.**

The system is designed to be modular, scalable, and easy to use. Start with the individual component tests, then move to the integrated system and web dashboard for full functionality.

