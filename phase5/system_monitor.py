#!/usr/bin/env python3
"""
CortexOS Phase 5: System Monitor
Comprehensive system monitoring and observability platform
"""

import asyncio
import logging
import time
import threading
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """System monitoring levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"

class ComponentStatus(Enum):
    """Component status types"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

@dataclass
class SystemMetric:
    """Individual system metric"""
    metric_id: str
    name: str
    metric_type: MetricType
    value: Union[int, float]
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentHealth:
    """Component health status"""
    component_id: str
    component_name: str
    status: ComponentStatus
    last_check: datetime
    uptime: float
    error_count: int = 0
    warning_count: int = 0
    metrics: Dict[str, SystemMetric] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_score: float = 1.0

@dataclass
class SystemSnapshot:
    """Complete system snapshot"""
    snapshot_id: str
    timestamp: datetime
    system_metrics: Dict[str, SystemMetric]
    component_health: Dict[str, ComponentHealth]
    overall_status: ComponentStatus
    overall_health_score: float
    active_alerts: int = 0
    performance_summary: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""
    monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED
    collection_interval: int = 30  # seconds
    retention_period: int = 86400  # 24 hours in seconds
    enable_system_metrics: bool = True
    enable_component_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_resource_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    """System metrics collection engine"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = defaultdict(deque)
        self.collection_start_time = time.time()
        
    def collect_system_metrics(self) -> Dict[str, SystemMetric]:
        """Collect comprehensive system metrics"""
        metrics = {}
        timestamp = datetime.now()
        
        try:
            # CPU Metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics['cpu_usage'] = SystemMetric(
                metric_id='cpu_usage',
                name='CPU Usage Percentage',
                metric_type=MetricType.GAUGE,
                value=cpu_percent,
                unit='percent',
                timestamp=timestamp
            )
            
            metrics['cpu_count'] = SystemMetric(
                metric_id='cpu_count',
                name='CPU Core Count',
                metric_type=MetricType.GAUGE,
                value=cpu_count,
                unit='cores',
                timestamp=timestamp
            )
            
            if cpu_freq:
                metrics['cpu_frequency'] = SystemMetric(
                    metric_id='cpu_frequency',
                    name='CPU Frequency',
                    metric_type=MetricType.GAUGE,
                    value=cpu_freq.current,
                    unit='MHz',
                    timestamp=timestamp
                )
            
            # Memory Metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics['memory_usage'] = SystemMetric(
                metric_id='memory_usage',
                name='Memory Usage Percentage',
                metric_type=MetricType.GAUGE,
                value=memory.percent,
                unit='percent',
                timestamp=timestamp
            )
            
            metrics['memory_total'] = SystemMetric(
                metric_id='memory_total',
                name='Total Memory',
                metric_type=MetricType.GAUGE,
                value=memory.total / (1024**3),  # GB
                unit='GB',
                timestamp=timestamp
            )
            
            metrics['memory_available'] = SystemMetric(
                metric_id='memory_available',
                name='Available Memory',
                metric_type=MetricType.GAUGE,
                value=memory.available / (1024**3),  # GB
                unit='GB',
                timestamp=timestamp
            )
            
            metrics['swap_usage'] = SystemMetric(
                metric_id='swap_usage',
                name='Swap Usage Percentage',
                metric_type=MetricType.GAUGE,
                value=swap.percent,
                unit='percent',
                timestamp=timestamp
            )
            
            # Disk Metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics['disk_usage'] = SystemMetric(
                metric_id='disk_usage',
                name='Disk Usage Percentage',
                metric_type=MetricType.GAUGE,
                value=(disk_usage.used / disk_usage.total) * 100,
                unit='percent',
                timestamp=timestamp
            )
            
            metrics['disk_total'] = SystemMetric(
                metric_id='disk_total',
                name='Total Disk Space',
                metric_type=MetricType.GAUGE,
                value=disk_usage.total / (1024**3),  # GB
                unit='GB',
                timestamp=timestamp
            )
            
            if disk_io:
                metrics['disk_read_bytes'] = SystemMetric(
                    metric_id='disk_read_bytes',
                    name='Disk Read Bytes',
                    metric_type=MetricType.COUNTER,
                    value=disk_io.read_bytes,
                    unit='bytes',
                    timestamp=timestamp
                )
                
                metrics['disk_write_bytes'] = SystemMetric(
                    metric_id='disk_write_bytes',
                    name='Disk Write Bytes',
                    metric_type=MetricType.COUNTER,
                    value=disk_io.write_bytes,
                    unit='bytes',
                    timestamp=timestamp
                )
            
            # Network Metrics
            network_io = psutil.net_io_counters()
            
            if network_io:
                metrics['network_bytes_sent'] = SystemMetric(
                    metric_id='network_bytes_sent',
                    name='Network Bytes Sent',
                    metric_type=MetricType.COUNTER,
                    value=network_io.bytes_sent,
                    unit='bytes',
                    timestamp=timestamp
                )
                
                metrics['network_bytes_recv'] = SystemMetric(
                    metric_id='network_bytes_recv',
                    name='Network Bytes Received',
                    metric_type=MetricType.COUNTER,
                    value=network_io.bytes_recv,
                    unit='bytes',
                    timestamp=timestamp
                )
            
            # Process Metrics
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            metrics['process_count'] = SystemMetric(
                metric_id='process_count',
                name='Total Process Count',
                metric_type=MetricType.GAUGE,
                value=process_count,
                unit='processes',
                timestamp=timestamp
            )
            
            metrics['cortexos_memory'] = SystemMetric(
                metric_id='cortexos_memory',
                name='CortexOS Memory Usage',
                metric_type=MetricType.GAUGE,
                value=current_process.memory_info().rss / (1024**2),  # MB
                unit='MB',
                timestamp=timestamp
            )
            
            metrics['cortexos_cpu'] = SystemMetric(
                metric_id='cortexos_cpu',
                name='CortexOS CPU Usage',
                metric_type=MetricType.GAUGE,
                value=current_process.cpu_percent(),
                unit='percent',
                timestamp=timestamp
            )
            
            # System Load
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                metrics['load_average_1m'] = SystemMetric(
                    metric_id='load_average_1m',
                    name='Load Average 1 Minute',
                    metric_type=MetricType.GAUGE,
                    value=load_avg[0],
                    unit='load',
                    timestamp=timestamp
                )
            
            # System Uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            metrics['system_uptime'] = SystemMetric(
                metric_id='system_uptime',
                name='System Uptime',
                metric_type=MetricType.GAUGE,
                value=uptime,
                unit='seconds',
                timestamp=timestamp
            )
            
            # Store metrics in history
            for metric_id, metric in metrics.items():
                self.metrics_history[metric_id].append(metric)
                
                # Maintain retention period
                cutoff_time = timestamp - timedelta(seconds=self.config.retention_period)
                while (self.metrics_history[metric_id] and 
                       self.metrics_history[metric_id][0].timestamp < cutoff_time):
                    self.metrics_history[metric_id].popleft()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_metric_history(self, metric_id: str, duration: int = 3600) -> List[SystemMetric]:
        """Get metric history for specified duration"""
        if metric_id not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=duration)
        return [metric for metric in self.metrics_history[metric_id] 
                if metric.timestamp >= cutoff_time]
    
    def calculate_metric_statistics(self, metric_id: str, duration: int = 3600) -> Dict[str, float]:
        """Calculate statistics for a metric over time"""
        history = self.get_metric_history(metric_id, duration)
        
        if not history:
            return {}
        
        values = [metric.value for metric in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'count': len(values),
            'latest': values[-1] if values else 0
        }

class ComponentMonitor:
    """Individual component monitoring"""
    
    def __init__(self, component_id: str, component_name: str):
        self.component_id = component_id
        self.component_name = component_name
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.warning_count = 0
        self.custom_metrics = {}
        self.health_checks = []
        
    def heartbeat(self):
        """Record component heartbeat"""
        self.last_heartbeat = time.time()
    
    def record_error(self, error_message: str = None):
        """Record component error"""
        self.error_count += 1
        logger.error(f"Component {self.component_id} error: {error_message}")
    
    def record_warning(self, warning_message: str = None):
        """Record component warning"""
        self.warning_count += 1
        logger.warning(f"Component {self.component_id} warning: {warning_message}")
    
    def add_custom_metric(self, metric: SystemMetric):
        """Add custom component metric"""
        self.custom_metrics[metric.metric_id] = metric
    
    def add_health_check(self, check_function: Callable[[], bool], check_name: str):
        """Add custom health check"""
        self.health_checks.append({
            'name': check_name,
            'function': check_function
        })
    
    def get_health_status(self) -> ComponentHealth:
        """Get current component health status"""
        current_time = time.time()
        uptime = current_time - self.start_time
        time_since_heartbeat = current_time - self.last_heartbeat
        
        # Determine status
        if time_since_heartbeat > 300:  # 5 minutes
            status = ComponentStatus.OFFLINE
            health_score = 0.0
        elif self.error_count > 10:
            status = ComponentStatus.CRITICAL
            health_score = 0.2
        elif self.error_count > 5 or self.warning_count > 20:
            status = ComponentStatus.WARNING
            health_score = 0.6
        else:
            status = ComponentStatus.HEALTHY
            health_score = 1.0
        
        # Run custom health checks
        for health_check in self.health_checks:
            try:
                if not health_check['function']():
                    status = ComponentStatus.WARNING
                    health_score = min(health_score, 0.7)
            except Exception as e:
                logger.error(f"Health check {health_check['name']} failed: {e}")
                status = ComponentStatus.WARNING
                health_score = min(health_score, 0.5)
        
        return ComponentHealth(
            component_id=self.component_id,
            component_name=self.component_name,
            status=status,
            last_check=datetime.now(),
            uptime=uptime,
            error_count=self.error_count,
            warning_count=self.warning_count,
            metrics=self.custom_metrics.copy(),
            health_score=health_score
        )

class SystemMonitor:
    """Comprehensive system monitoring platform"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.component_monitors = {}
        self.system_snapshots = deque()
        self.running = False
        self.monitor_task = None
        self.alert_callbacks = []
        
        logger.info("System Monitor initialized")
    
    async def start(self):
        """Start system monitoring"""
        try:
            self.running = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("System Monitor started")
            
        except Exception as e:
            logger.error(f"Error starting System Monitor: {e}")
            raise
    
    async def stop(self):
        """Stop system monitoring"""
        try:
            self.running = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("System Monitor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping System Monitor: {e}")
    
    def register_component(self, component_id: str, component_name: str) -> ComponentMonitor:
        """Register component for monitoring"""
        monitor = ComponentMonitor(component_id, component_name)
        self.component_monitors[component_id] = monitor
        logger.info(f"Registered component for monitoring: {component_id}")
        return monitor
    
    def unregister_component(self, component_id: str):
        """Unregister component from monitoring"""
        if component_id in self.component_monitors:
            del self.component_monitors[component_id]
            logger.info(f"Unregistered component from monitoring: {component_id}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.running:
            try:
                # Collect system metrics
                system_metrics = {}
                if self.config.enable_system_metrics:
                    system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Collect component health
                component_health = {}
                if self.config.enable_component_monitoring:
                    for component_id, monitor in self.component_monitors.items():
                        component_health[component_id] = monitor.get_health_status()
                
                # Create system snapshot
                snapshot = self._create_system_snapshot(system_metrics, component_health)
                
                # Store snapshot
                self.system_snapshots.append(snapshot)
                
                # Maintain retention
                cutoff_time = datetime.now() - timedelta(seconds=self.config.retention_period)
                while (self.system_snapshots and 
                       self.system_snapshots[0].timestamp < cutoff_time):
                    self.system_snapshots.popleft()
                
                # Check for alerts
                await self._check_alerts(snapshot)
                
                # Wait for next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
        
        logger.info("Monitoring loop stopped")
    
    def _create_system_snapshot(self, system_metrics: Dict[str, SystemMetric], 
                               component_health: Dict[str, ComponentHealth]) -> SystemSnapshot:
        """Create comprehensive system snapshot"""
        timestamp = datetime.now()
        
        # Calculate overall health score
        health_scores = []
        critical_components = 0
        warning_components = 0
        
        for health in component_health.values():
            health_scores.append(health.health_score)
            if health.status == ComponentStatus.CRITICAL:
                critical_components += 1
            elif health.status == ComponentStatus.WARNING:
                warning_components += 1
        
        # Overall health score
        if health_scores:
            overall_health_score = sum(health_scores) / len(health_scores)
        else:
            overall_health_score = 1.0
        
        # Overall status
        if critical_components > 0:
            overall_status = ComponentStatus.CRITICAL
        elif warning_components > 0:
            overall_status = ComponentStatus.WARNING
        else:
            overall_status = ComponentStatus.HEALTHY
        
        # Performance summary
        performance_summary = {}
        if 'cpu_usage' in system_metrics:
            performance_summary['cpu_usage'] = system_metrics['cpu_usage'].value
        if 'memory_usage' in system_metrics:
            performance_summary['memory_usage'] = system_metrics['memory_usage'].value
        if 'disk_usage' in system_metrics:
            performance_summary['disk_usage'] = system_metrics['disk_usage'].value
        
        return SystemSnapshot(
            snapshot_id=f"snapshot_{int(time.time() * 1000000)}",
            timestamp=timestamp,
            system_metrics=system_metrics,
            component_health=component_health,
            overall_status=overall_status,
            overall_health_score=overall_health_score,
            performance_summary=performance_summary
        )
    
    async def _check_alerts(self, snapshot: SystemSnapshot):
        """Check for alert conditions"""
        alerts = []
        
        # Check system metric thresholds
        for metric_id, metric in snapshot.system_metrics.items():
            if metric_id in self.config.alert_thresholds:
                threshold = self.config.alert_thresholds[metric_id]
                if metric.value > threshold:
                    alert = {
                        'type': 'metric_threshold',
                        'metric_id': metric_id,
                        'value': metric.value,
                        'threshold': threshold,
                        'timestamp': metric.timestamp
                    }
                    alerts.append(alert)
        
        # Check component health
        for component_id, health in snapshot.component_health.items():
            if health.status in [ComponentStatus.CRITICAL, ComponentStatus.OFFLINE]:
                alert = {
                    'type': 'component_health',
                    'component_id': component_id,
                    'status': health.status.value,
                    'health_score': health.health_score,
                    'timestamp': health.last_check
                }
                alerts.append(alert)
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def get_current_snapshot(self) -> Optional[SystemSnapshot]:
        """Get most recent system snapshot"""
        return self.system_snapshots[-1] if self.system_snapshots else None
    
    def get_snapshot_history(self, duration: int = 3600) -> List[SystemSnapshot]:
        """Get snapshot history for specified duration"""
        cutoff_time = datetime.now() - timedelta(seconds=duration)
        return [snapshot for snapshot in self.system_snapshots 
                if snapshot.timestamp >= cutoff_time]
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get current health status for component"""
        if component_id in self.component_monitors:
            return self.component_monitors[component_id].get_health_status()
        return None
    
    def get_system_metrics(self) -> Dict[str, SystemMetric]:
        """Get current system metrics"""
        return self.metrics_collector.collect_system_metrics()
    
    def get_metric_statistics(self, metric_id: str, duration: int = 3600) -> Dict[str, float]:
        """Get metric statistics over time"""
        return self.metrics_collector.calculate_metric_statistics(metric_id, duration)
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        current_snapshot = self.get_current_snapshot()
        
        return {
            'running': self.running,
            'monitoring_level': self.config.monitoring_level.value,
            'collection_interval': self.config.collection_interval,
            'registered_components': len(self.component_monitors),
            'snapshots_stored': len(self.system_snapshots),
            'alert_callbacks': len(self.alert_callbacks),
            'current_status': current_snapshot.overall_status.value if current_snapshot else 'unknown',
            'current_health_score': current_snapshot.overall_health_score if current_snapshot else 0.0,
            'uptime': time.time() - self.metrics_collector.collection_start_time
        }

# Test and demonstration
async def test_system_monitor():
    """Test the system monitor"""
    print("🧠 Testing CortexOS System Monitor...")
    
    # Create monitoring configuration
    config = MonitoringConfig(
        monitoring_level=MonitoringLevel.COMPREHENSIVE,
        collection_interval=5,  # 5 seconds for testing
        retention_period=300,   # 5 minutes for testing
        enable_system_metrics=True,
        enable_component_monitoring=True,
        alert_thresholds={
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0
        }
    )
    
    # Initialize monitor
    monitor = SystemMonitor(config)
    
    # Add alert callback
    def alert_handler(alert_type: str, alert_data: Dict[str, Any]):
        print(f"🚨 ALERT [{alert_type}]: {alert_data}")
    
    monitor.add_alert_callback(alert_handler)
    
    try:
        # Start monitoring
        await monitor.start()
        print("✅ System Monitor started")
        
        # Register test components
        neural_engine = monitor.register_component("neural_engine", "Neural Processing Engine")
        memory_system = monitor.register_component("memory_system", "Memory Management System")
        
        # Add custom health checks
        def neural_engine_check():
            return True  # Simulate healthy component
        
        def memory_system_check():
            return True  # Simulate healthy component
        
        neural_engine.add_health_check(neural_engine_check, "Neural Engine Health")
        memory_system.add_health_check(memory_system_check, "Memory System Health")
        
        # Simulate component activity
        print("\n📊 Simulating component activity...")
        for i in range(3):
            neural_engine.heartbeat()
            memory_system.heartbeat()
            
            # Add custom metrics
            neural_engine.add_custom_metric(SystemMetric(
                metric_id=f"neural_throughput_{i}",
                name="Neural Processing Throughput",
                metric_type=MetricType.GAUGE,
                value=100 + i * 10,
                unit="ops/sec",
                timestamp=datetime.now()
            ))
            
            await asyncio.sleep(2)
        
        # Wait for monitoring data collection
        print("⏳ Collecting monitoring data...")
        await asyncio.sleep(8)
        
        # Get current snapshot
        current_snapshot = monitor.get_current_snapshot()
        if current_snapshot:
            print(f"\n📈 Current System Status:")
            print(f"   Overall Status: {current_snapshot.overall_status.value}")
            print(f"   Health Score: {current_snapshot.overall_health_score:.3f}")
            print(f"   Active Components: {len(current_snapshot.component_health)}")
            
            # Display key metrics
            print(f"\n💻 System Metrics:")
            for metric_id, metric in current_snapshot.system_metrics.items():
                if metric_id in ['cpu_usage', 'memory_usage', 'disk_usage']:
                    print(f"   {metric.name}: {metric.value:.1f} {metric.unit}")
            
            # Display component health
            print(f"\n🔧 Component Health:")
            for comp_id, health in current_snapshot.component_health.items():
                print(f"   {health.component_name}: {health.status.value} (Score: {health.health_score:.3f})")
                print(f"      Uptime: {health.uptime:.1f}s, Errors: {health.error_count}")
        
        # Get metric statistics
        print(f"\n📊 Metric Statistics (last 5 minutes):")
        cpu_stats = monitor.get_metric_statistics('cpu_usage', 300)
        if cpu_stats:
            print(f"   CPU Usage - Min: {cpu_stats['min']:.1f}%, Max: {cpu_stats['max']:.1f}%, Avg: {cpu_stats['mean']:.1f}%")
        
        memory_stats = monitor.get_metric_statistics('memory_usage', 300)
        if memory_stats:
            print(f"   Memory Usage - Min: {memory_stats['min']:.1f}%, Max: {memory_stats['max']:.1f}%, Avg: {memory_stats['mean']:.1f}%")
        
        # Display monitoring status
        print(f"\n🔧 Monitor Status:")
        status = monitor.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n✅ System Monitor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(test_system_monitor())

