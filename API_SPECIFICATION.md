# CortexOS API Specification v1.0

## Overview

This document defines the **locked and frozen** API interfaces for all CortexOS components. All components MUST implement these interfaces through the `compat_api.py` compatibility layer.

**Status**: LOCKED ✅  
**Version**: 1.0.0  
**Last Updated**: October 28, 2025  

## Design Principles

1. **Consistency**: All similar operations use the same method names
2. **Simplicity**: Simple operations have simple APIs
3. **Compatibility**: Backward compatibility through wrapper layer
4. **Discoverability**: Method names are self-documenting

## Core Interfaces

### Infrastructure Layer

#### CubeStorage

```python
class CubeStorage:
    """Key-value storage with advanced cube features"""
    
    def __init__(self, storage_path: str = None, **kwargs) -> None
    def store(self, key: str, value: Any) -> bool
    def retrieve(self, key: str) -> Any
    def delete(self, key: str) -> bool
    def list_keys(self) -> List[str]
```

#### ContractManager

```python
class ContractManager:
    """Manage contracts and agreements"""
    
    def __init__(self, **kwargs) -> None
    def register_contract(self, name: str, contract: Dict) -> bool
    def get_contract(self, name: str) -> Dict
    def list_contracts(self) -> List[str]
```

#### NeuralFabric

```python
class NeuralFabric:
    """Neural signal routing and processing"""
    
    def __init__(self, **kwargs) -> None
    def route_signal(self, signal: Dict) -> Any
    def get_status(self) -> Dict
```

#### GPUManager

```python
class GPUManager:
    """GPU resource management"""
    
    def __init__(self, enable_gpu: bool = True, **kwargs) -> None
    def get_device(self, index: int = 0) -> Any
    def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]
    def get_all_gpu_info(self) -> List[GPUInfo]
    def clear_cache(self) -> None
```

### Phase 1: Neural Processing

#### NeuroEngine

```python
class NeuroEngine:
    """Core neural processing engine"""
    
    def __init__(self, **kwargs) -> None
    def process(self, data: Union[np.ndarray, List, Any], **kwargs) -> Any
    def start(self) -> bool
    def stop(self) -> bool
    def get_engine_stats(self) -> Dict
```

#### ContextEngine

```python
class ContextEngine:
    """Context management"""
    
    def __init__(self, **kwargs) -> None
    def set_context(self, context: Dict[str, Any]) -> bool
    def get_context(self) -> Dict[str, Any]
    def clear_context(self) -> bool
```

#### NeuralGatekeeper

```python
class NeuralGatekeeper:
    """Input validation and gating"""
    
    def __init__(self, **kwargs) -> None
    def validate(self, data: Any) -> bool
```

#### CortexVectorizer

```python
class CortexVectorizer:
    """Text and data vectorization"""
    
    def __init__(self, **kwargs) -> None
    def vectorize(self, text: str, input_type: str = "text", **kwargs) -> np.ndarray
```

### Phase 2: Resonance

#### ResonanceField

```python
class ResonanceField:
    """Resonance field management"""
    
    def __init__(self, **kwargs) -> None
    def update(self, data: np.ndarray) -> bool
    def query(self, data: np.ndarray) -> Any
```

#### ResonanceMonitor

```python
class ResonanceMonitor:
    """Monitor resonance patterns"""
    
    def __init__(self, **kwargs) -> None
    def get_status(self) -> Dict[str, Any]
```

#### ResonanceReinforcer

```python
class ResonanceReinforcer:
    """Reinforce resonance patterns"""
    
    def __init__(self, **kwargs) -> None
    def reinforce(self, pattern: np.ndarray) -> np.ndarray
```

#### TopKSparseResonance

```python
class TopKSparseResonance:
    """Top-K sparse resonance selection"""
    
    def __init__(self, k: int = 5, **kwargs) -> None
    def select_top_k(self, data: np.ndarray) -> List
```

### Phase 3: Memory

#### MemoryInserter

```python
class MemoryInserter:
    """Insert memories into the system"""
    
    def __init__(self, **kwargs) -> None
    def insert(self, memory: Dict[str, Any]) -> bool
```

#### MemoryRetriever

```python
class MemoryRetriever:
    """Retrieve memories from the system"""
    
    def __init__(self, **kwargs) -> None
    def retrieve(self, query: str, limit: int = 10) -> List[Dict]
    def get_stats(self) -> Dict[str, Any]
```

#### MemoryConsolidator

```python
class MemoryConsolidator:
    """Consolidate and optimize memories"""
    
    def __init__(self, **kwargs) -> None
    def consolidate(self) -> Dict[str, Any]
```

#### CognitiveBridge

```python
class CognitiveBridge:
    """Bridge neural and memory systems"""
    
    def __init__(self, **kwargs) -> None
    def bridge(self, neural_data: Any, memory_data: Any) -> Dict[str, Any]
```

### Phase 4: Data Ingestion

#### DataIngestionEngine

```python
class DataIngestionEngine:
    """Ingest data into the system"""
    
    def __init__(self, **kwargs) -> None
    def ingest(self, data: Dict[str, Any]) -> bool
```

#### StreamProcessor

```python
class StreamProcessor:
    """Process streaming data"""
    
    def __init__(self, config: Optional[StreamConfig] = None, **kwargs) -> None
    def process(self, data: Any) -> Any
```

#### BatchProcessor

```python
class BatchProcessor:
    """Process batch data"""
    
    def __init__(self, config: Optional[BatchConfig] = None, **kwargs) -> None
    def process_batch(self, batch: List[Dict]) -> Dict[str, Any]
```

#### IngestionValidator

```python
class IngestionValidator:
    """Validate ingestion data"""
    
    def __init__(self, **kwargs) -> None
    def validate(self, data: Any) -> bool
```

### Phase 5: Monitoring

#### SystemMonitor

```python
class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self, **kwargs) -> None
    def get_system_status(self) -> Dict[str, Any]
```

#### PerformanceTracker

```python
class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self, **kwargs) -> None
    def track_metric(self, name: str, value: float) -> bool
    def get_metrics(self) -> Dict[str, Any]
```

#### HealthChecker

```python
class HealthChecker:
    """Check system health"""
    
    def __init__(self, **kwargs) -> None
    def check_health(self) -> Dict[str, Any]
```

#### AlertManager

```python
class AlertManager:
    """Manage system alerts"""
    
    def __init__(self, **kwargs) -> None
    def create_alert(self, alert: Dict[str, Any]) -> bool
    def get_alerts(self) -> List[Dict]
```

### Phase 6: Optimization

#### MoodModulator

```python
class MoodModulator:
    """Modulate system mood"""
    
    def __init__(self, **kwargs) -> None
    def modulate(self, mood: Dict[str, float]) -> Dict[str, Any]
```

#### CognitiveEnhancer

```python
class CognitiveEnhancer:
    """Enhance cognitive processing"""
    
    def __init__(self, **kwargs) -> None
    def enhance(self, data: Any) -> Any
```

#### NeuralOptimizer

```python
class NeuralOptimizer:
    """Optimize neural parameters"""
    
    def __init__(self, **kwargs) -> None
    def optimize(self, params: Dict[str, Any]) -> Dict[str, Any]
    def get_history(self) -> List[Dict]
```

#### AdaptiveController

```python
class AdaptiveController:
    """Adaptive system control"""
    
    def __init__(self, **kwargs) -> None
    def adapt(self, metrics: Dict[str, Any]) -> Dict[str, Any]
```

## REST API Endpoints

All components are accessible via REST API at `http://localhost:8080/api/`

### System Endpoints

- `GET /health` - Health check
- `GET /api/status` - System status
- `GET /metrics` - Prometheus metrics

### Component Endpoints

- `GET /api/components` - List all components
- `GET /api/components/{phase}/{component}` - Component details

### Memory Endpoints

- `GET /api/memory/stats` - Memory statistics
- `POST /api/memory/insert` - Insert memory
- `POST /api/memory/search` - Search memories
- `POST /api/memory/consolidate` - Consolidate memories

### Neural Endpoints

- `POST /api/neural/process` - Process neural data
- `GET /api/neural/performance` - Performance metrics
- `POST /api/neural/optimization/request` - Request optimization
- `GET /api/neural/optimization/history` - Optimization history

### Data Ingestion Endpoints

- `POST /api/ingest/single` - Ingest single item
- `POST /api/ingest/batch` - Batch ingestion

### GPU Endpoints

- `GET /api/gpu/status` - GPU status
- `GET /api/gpu/metrics` - GPU metrics

### Monitoring Endpoints

- `GET /api/monitoring/metrics` - System metrics
- `GET /api/monitoring/alerts` - System alerts

### Utility Endpoints

- `POST /api/vectorize` - Vectorize text
- `POST /api/context/set` - Set context
- `GET /api/context/get` - Get context
- `GET /api/resonance/field` - Resonance field status
- `POST /api/mood/modulate` - Modulate mood

## Compatibility Layer

The `compat_api.py` module provides the compatibility layer that maps these standard interfaces to the actual implementation methods. This ensures:

1. **Consistent API** across all components
2. **Backward compatibility** with existing code
3. **Forward compatibility** for future changes
4. **Easy testing** with predictable interfaces

## Usage

### Python API

```python
# Import from compatibility layer
from compat_api import (
    NeuroEngine,
    MemoryInserter,
    MemoryRetriever,
    CubeStorage
)

# Use consistent interfaces
engine = NeuroEngine()
result = engine.process(data)

memory = MemoryInserter()
memory.insert({"content": "test"})

storage = CubeStorage()
storage.store("key", "value")
value = storage.retrieve("key")
```

### REST API

```bash
# Health check
curl http://localhost:8080/health

# Insert memory
curl -X POST http://localhost:8080/api/memory/insert \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory"}'

# Process neural data
curl -X POST http://localhost:8080/api/neural/process \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3, 4, 5]}'
```

## Versioning

This API specification follows semantic versioning:

- **Major version** (1.x.x): Breaking changes to interfaces
- **Minor version** (x.1.x): New features, backward compatible
- **Patch version** (x.x.1): Bug fixes, backward compatible

**Current Version**: 1.0.0

## Change Policy

1. **No breaking changes** without major version bump
2. **All changes** must update this specification
3. **Deprecation period** of 6 months for removed features
4. **Compatibility layer** must support at least 2 major versions

## Testing

All components MUST pass the compatibility test suite:

```bash
python3 tests/test_with_compat_api.py
```

Expected result: **100% pass rate** (25/25 tests)

## Compliance

To be compliant with this specification, a component must:

1. ✅ Implement all required methods
2. ✅ Accept all specified parameters
3. ✅ Return values in specified formats
4. ✅ Pass all compatibility tests
5. ✅ Be accessible via REST API

## Support

For questions or issues with this specification:

- GitHub Issues: https://github.com/your-org/cortexos/issues
- Documentation: See `DEPLOYMENT_GUIDE.md`
- API Server: See `api_server.py`
- Compatibility Layer: See `compat_api.py`

---

**Document Status**: LOCKED ✅  
**Effective Date**: October 28, 2025  
**Next Review**: April 28, 2026
