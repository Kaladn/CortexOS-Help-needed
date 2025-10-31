#!/usr/bin/env python3
"""
CortexOS Phase 5: Health Checker
Comprehensive system health monitoring and diagnostic system
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
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

class CheckType(Enum):
    """Types of health checks"""
    SYSTEM = "system"
    COMPONENT = "component"
    DEPENDENCY = "dependency"
    CUSTOM = "custom"
    INTEGRATION = "integration"

class CheckSeverity(Enum):
    """Health check severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    check_id: str
    name: str
    description: str
    check_type: CheckType
    severity: CheckSeverity
    check_function: Callable[[], Tuple[HealthStatus, str, Dict[str, Any]]]
    interval: int = 60  # seconds
    timeout: int = 30   # seconds
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class HealthCheckResult:
    """Result of a health check execution"""
    check_id: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float
    error: Optional[str] = None

@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_id: str
    component_name: str
    overall_status: HealthStatus
    health_score: float  # 0.0 to 1.0
    check_results: Dict[str, HealthCheckResult]
    last_updated: datetime
    uptime: float
    error_count: int = 0
    warning_count: int = 0

@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_status: HealthStatus
    overall_score: float
    component_health: Dict[str, ComponentHealth]
    failed_checks: List[str]
    warning_checks: List[str]
    total_checks: int
    passed_checks: int
    timestamp: datetime
    health_summary: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthTrend:
    """Health trend analysis"""
    component_id: str
    trend_direction: str  # "improving", "stable", "degrading"
    trend_strength: float
    health_history: List[float]
    prediction: Optional[float] = None

class SystemHealthChecks:
    """Built-in system health checks"""
    
    @staticmethod
    def check_cpu_health() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check CPU health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            details = {
                'cpu_usage': cpu_percent,
                'cpu_count': cpu_count,
                'load_average_1m': load_avg[0],
                'load_average_5m': load_avg[1],
                'load_average_15m': load_avg[2]
            }
            
            if cpu_percent > 95:
                return HealthStatus.CRITICAL, f"CPU usage critically high: {cpu_percent:.1f}%", details
            elif cpu_percent > 85:
                return HealthStatus.WARNING, f"CPU usage high: {cpu_percent:.1f}%", details
            else:
                return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check CPU health: {e}", {}
    
    @staticmethod
    def check_memory_health() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory health"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            details = {
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent
            }
            
            if memory.percent > 95:
                return HealthStatus.CRITICAL, f"Memory usage critically high: {memory.percent:.1f}%", details
            elif memory.percent > 85:
                return HealthStatus.WARNING, f"Memory usage high: {memory.percent:.1f}%", details
            else:
                return HealthStatus.HEALTHY, f"Memory usage normal: {memory.percent:.1f}%", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check memory health: {e}", {}
    
    @staticmethod
    def check_disk_health() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check disk health"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            details = {
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_percent': usage_percent
            }
            
            if disk_io:
                details.update({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                })
            
            if usage_percent > 95:
                return HealthStatus.CRITICAL, f"Disk usage critically high: {usage_percent:.1f}%", details
            elif usage_percent > 85:
                return HealthStatus.WARNING, f"Disk usage high: {usage_percent:.1f}%", details
            else:
                return HealthStatus.HEALTHY, f"Disk usage normal: {usage_percent:.1f}%", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check disk health: {e}", {}
    
    @staticmethod
    def check_network_health() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check network health"""
        try:
            network_io = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            details = {
                'connections_count': network_connections
            }
            
            if network_io:
                details.update({
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv,
                    'errin': network_io.errin,
                    'errout': network_io.errout,
                    'dropin': network_io.dropin,
                    'dropout': network_io.dropout
                })
                
                # Check for network errors
                total_errors = network_io.errin + network_io.errout
                total_drops = network_io.dropin + network_io.dropout
                
                if total_errors > 1000 or total_drops > 1000:
                    return HealthStatus.WARNING, f"Network errors detected: {total_errors} errors, {total_drops} drops", details
            
            if network_connections > 10000:
                return HealthStatus.WARNING, f"High number of network connections: {network_connections}", details
            else:
                return HealthStatus.HEALTHY, f"Network health normal: {network_connections} connections", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check network health: {e}", {}
    
    @staticmethod
    def check_process_health() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check process health"""
        try:
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            
            details = {
                'total_processes': process_count,
                'cortexos_pid': current_process.pid,
                'cortexos_memory_mb': current_process.memory_info().rss / (1024 * 1024),
                'cortexos_cpu_percent': current_process.cpu_percent(),
                'cortexos_threads': current_process.num_threads(),
                'cortexos_status': current_process.status()
            }
            
            # Check for zombie processes
            zombie_count = 0
            for proc in psutil.process_iter(['status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            details['zombie_processes'] = zombie_count
            
            if zombie_count > 10:
                return HealthStatus.WARNING, f"High number of zombie processes: {zombie_count}", details
            elif process_count > 1000:
                return HealthStatus.WARNING, f"High number of processes: {process_count}", details
            else:
                return HealthStatus.HEALTHY, f"Process health normal: {process_count} processes", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check process health: {e}", {}
    
    @staticmethod
    def check_file_descriptors() -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check file descriptor usage"""
        try:
            current_process = psutil.Process()
            fd_count = current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
            
            # Get system limits
            try:
                import resource
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            except ImportError:
                soft_limit, hard_limit = 1024, 4096
            
            details = {
                'file_descriptors': fd_count,
                'soft_limit': soft_limit,
                'hard_limit': hard_limit,
                'usage_percent': (fd_count / soft_limit) * 100 if soft_limit > 0 else 0
            }
            
            usage_percent = details['usage_percent']
            
            if usage_percent > 90:
                return HealthStatus.CRITICAL, f"File descriptor usage critically high: {usage_percent:.1f}%", details
            elif usage_percent > 75:
                return HealthStatus.WARNING, f"File descriptor usage high: {usage_percent:.1f}%", details
            else:
                return HealthStatus.HEALTHY, f"File descriptor usage normal: {usage_percent:.1f}%", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check file descriptors: {e}", {}

class ComponentHealthChecks:
    """Component-specific health checks"""
    
    @staticmethod
    def check_component_responsiveness(component_id: str, ping_function: Callable = None) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check if component is responsive"""
        try:
            start_time = time.time()
            
            if ping_function:
                response = ping_function()
                response_time = time.time() - start_time
                
                details = {
                    'response_time': response_time,
                    'response': response
                }
                
                if response_time > 5.0:
                    return HealthStatus.WARNING, f"Component {component_id} slow response: {response_time:.3f}s", details
                else:
                    return HealthStatus.HEALTHY, f"Component {component_id} responsive: {response_time:.3f}s", details
            else:
                return HealthStatus.UNKNOWN, f"No ping function provided for {component_id}", {}
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Component {component_id} unresponsive: {e}", {}
    
    @staticmethod
    def check_component_memory_usage(component_id: str, memory_tracker: Callable = None) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check component memory usage"""
        try:
            if memory_tracker:
                memory_usage = memory_tracker()
                
                details = {
                    'memory_usage_mb': memory_usage,
                    'component_id': component_id
                }
                
                if memory_usage > 1000:  # 1GB
                    return HealthStatus.WARNING, f"Component {component_id} high memory usage: {memory_usage:.1f}MB", details
                else:
                    return HealthStatus.HEALTHY, f"Component {component_id} memory usage normal: {memory_usage:.1f}MB", details
            else:
                return HealthStatus.UNKNOWN, f"No memory tracker provided for {component_id}", {}
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check memory for {component_id}: {e}", {}
    
    @staticmethod
    def check_component_error_rate(component_id: str, error_counter: Callable = None) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check component error rate"""
        try:
            if error_counter:
                error_count, total_operations = error_counter()
                error_rate = (error_count / total_operations) * 100 if total_operations > 0 else 0
                
                details = {
                    'error_count': error_count,
                    'total_operations': total_operations,
                    'error_rate_percent': error_rate,
                    'component_id': component_id
                }
                
                if error_rate > 10:
                    return HealthStatus.CRITICAL, f"Component {component_id} high error rate: {error_rate:.1f}%", details
                elif error_rate > 5:
                    return HealthStatus.WARNING, f"Component {component_id} elevated error rate: {error_rate:.1f}%", details
                else:
                    return HealthStatus.HEALTHY, f"Component {component_id} error rate normal: {error_rate:.1f}%", details
            else:
                return HealthStatus.UNKNOWN, f"No error counter provided for {component_id}", {}
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Failed to check error rate for {component_id}: {e}", {}

class HealthChecker:
    """Comprehensive system health monitoring and diagnostic system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.health_checks = {}
        self.component_health = {}
        self.health_history = defaultdict(lambda: deque(maxlen=1000))
        self.check_results = {}
        
        # Configuration
        self.default_check_interval = self.config.get('default_check_interval', 60)
        self.enable_trend_analysis = self.config.get('enable_trend_analysis', True)
        self.health_retention_hours = self.config.get('health_retention_hours', 24)
        self.parallel_checks = self.config.get('parallel_checks', True)
        self.max_concurrent_checks = self.config.get('max_concurrent_checks', 10)
        
        # State
        self.running = False
        self.check_tasks = {}
        self.health_monitor_task = None
        
        # Register built-in system checks
        self._register_builtin_checks()
        
        logger.info("Health Checker initialized")
    
    def _register_builtin_checks(self):
        """Register built-in system health checks"""
        builtin_checks = [
            HealthCheck(
                check_id="system_cpu",
                name="CPU Health Check",
                description="Monitor CPU usage and load",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.HIGH,
                check_function=SystemHealthChecks.check_cpu_health,
                interval=30
            ),
            HealthCheck(
                check_id="system_memory",
                name="Memory Health Check",
                description="Monitor memory and swap usage",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.HIGH,
                check_function=SystemHealthChecks.check_memory_health,
                interval=30
            ),
            HealthCheck(
                check_id="system_disk",
                name="Disk Health Check",
                description="Monitor disk usage and I/O",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.MEDIUM,
                check_function=SystemHealthChecks.check_disk_health,
                interval=60
            ),
            HealthCheck(
                check_id="system_network",
                name="Network Health Check",
                description="Monitor network connectivity and errors",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.MEDIUM,
                check_function=SystemHealthChecks.check_network_health,
                interval=60
            ),
            HealthCheck(
                check_id="system_processes",
                name="Process Health Check",
                description="Monitor process count and status",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.LOW,
                check_function=SystemHealthChecks.check_process_health,
                interval=120
            ),
            HealthCheck(
                check_id="system_file_descriptors",
                name="File Descriptor Check",
                description="Monitor file descriptor usage",
                check_type=CheckType.SYSTEM,
                severity=CheckSeverity.MEDIUM,
                check_function=SystemHealthChecks.check_file_descriptors,
                interval=300
            )
        ]
        
        for check in builtin_checks:
            self.register_health_check(check)
    
    async def start(self):
        """Start health monitoring"""
        try:
            self.running = True
            
            # Start individual check tasks
            for check_id, health_check in self.health_checks.items():
                if health_check.enabled:
                    task = asyncio.create_task(self._check_loop(health_check))
                    self.check_tasks[check_id] = task
            
            # Start health monitoring task
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            logger.info(f"Health Checker started with {len(self.check_tasks)} active checks")
            
        except Exception as e:
            logger.error(f"Error starting Health Checker: {e}")
            raise
    
    async def stop(self):
        """Stop health monitoring"""
        try:
            self.running = False
            
            # Cancel check tasks
            for task in self.check_tasks.values():
                task.cancel()
            
            # Cancel health monitor task
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Wait for tasks to complete
            all_tasks = list(self.check_tasks.values())
            if self.health_monitor_task:
                all_tasks.append(self.health_monitor_task)
            
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
            logger.info("Health Checker stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Health Checker: {e}")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.check_id] = health_check
        logger.info(f"Registered health check: {health_check.check_id}")
    
    def unregister_health_check(self, check_id: str):
        """Unregister a health check"""
        if check_id in self.health_checks:
            del self.health_checks[check_id]
            
            # Cancel running task if exists
            if check_id in self.check_tasks:
                self.check_tasks[check_id].cancel()
                del self.check_tasks[check_id]
            
            logger.info(f"Unregistered health check: {check_id}")
    
    def register_component(self, component_id: str, component_name: str, custom_checks: List[HealthCheck] = None):
        """Register a component for health monitoring"""
        # Initialize component health
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            component_name=component_name,
            overall_status=HealthStatus.UNKNOWN,
            health_score=0.0,
            check_results={},
            last_updated=datetime.now(),
            uptime=0.0
        )
        
        # Register custom checks if provided
        if custom_checks:
            for check in custom_checks:
                # Prefix check ID with component ID
                check.check_id = f"{component_id}_{check.check_id}"
                self.register_health_check(check)
        
        logger.info(f"Registered component for health monitoring: {component_id}")
    
    async def _check_loop(self, health_check: HealthCheck):
        """Individual health check loop"""
        logger.info(f"Health check loop started: {health_check.check_id}")
        
        while self.running:
            try:
                # Execute health check
                result = await self._execute_health_check(health_check)
                
                # Store result
                self.check_results[health_check.check_id] = result
                
                # Update component health if applicable
                await self._update_component_health(health_check.check_id, result)
                
                # Wait for next check
                await asyncio.sleep(health_check.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop {health_check.check_id}: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Health check loop stopped: {health_check.check_id}")
    
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check"""
        start_time = time.time()
        
        try:
            # Execute check function with timeout
            if asyncio.iscoroutinefunction(health_check.check_function):
                status, message, details = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout
                )
            else:
                status, message, details = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, health_check.check_function
                    ),
                    timeout=health_check.timeout
                )
            
            execution_time = time.time() - start_time
            
            return HealthCheckResult(
                check_id=health_check.check_id,
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return HealthCheckResult(
                check_id=health_check.check_id,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {health_check.timeout}s",
                details={},
                timestamp=datetime.now(),
                execution_time=execution_time,
                error="timeout"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HealthCheckResult(
                check_id=health_check.check_id,
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {e}",
                details={},
                timestamp=datetime.now(),
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _update_component_health(self, check_id: str, result: HealthCheckResult):
        """Update component health based on check result"""
        try:
            # Determine component ID from check ID
            component_id = None
            for comp_id in self.component_health.keys():
                if check_id.startswith(f"{comp_id}_") or check_id.startswith("system"):
                    component_id = comp_id if not check_id.startswith("system") else "system"
                    break
            
            # Create system component if needed
            if component_id == "system" and "system" not in self.component_health:
                self.component_health["system"] = ComponentHealth(
                    component_id="system",
                    component_name="System",
                    overall_status=HealthStatus.UNKNOWN,
                    health_score=0.0,
                    check_results={},
                    last_updated=datetime.now(),
                    uptime=time.time()
                )
                component_id = "system"
            
            if component_id and component_id in self.component_health:
                component = self.component_health[component_id]
                
                # Update check results
                component.check_results[check_id] = result
                component.last_updated = datetime.now()
                
                # Update error/warning counts
                if result.status == HealthStatus.CRITICAL:
                    component.error_count += 1
                elif result.status == HealthStatus.WARNING:
                    component.warning_count += 1
                
                # Calculate overall component health
                await self._calculate_component_health(component_id)
            
        except Exception as e:
            logger.error(f"Error updating component health: {e}")
    
    async def _calculate_component_health(self, component_id: str):
        """Calculate overall health for a component"""
        try:
            component = self.component_health[component_id]
            
            if not component.check_results:
                component.overall_status = HealthStatus.UNKNOWN
                component.health_score = 0.0
                return
            
            # Calculate health score based on check results
            status_weights = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.WARNING: 0.6,
                HealthStatus.DEGRADED: 0.4,
                HealthStatus.CRITICAL: 0.0,
                HealthStatus.UNKNOWN: 0.3
            }
            
            severity_weights = {
                CheckSeverity.CRITICAL: 1.0,
                CheckSeverity.HIGH: 0.8,
                CheckSeverity.MEDIUM: 0.6,
                CheckSeverity.LOW: 0.4
            }
            
            total_weight = 0.0
            weighted_score = 0.0
            critical_count = 0
            warning_count = 0
            
            for check_id, result in component.check_results.items():
                if check_id in self.health_checks:
                    check = self.health_checks[check_id]
                    severity_weight = severity_weights.get(check.severity, 0.5)
                    status_score = status_weights.get(result.status, 0.0)
                    
                    weighted_score += status_score * severity_weight
                    total_weight += severity_weight
                    
                    if result.status == HealthStatus.CRITICAL:
                        critical_count += 1
                    elif result.status == HealthStatus.WARNING:
                        warning_count += 1
            
            # Calculate final health score
            if total_weight > 0:
                component.health_score = weighted_score / total_weight
            else:
                component.health_score = 0.0
            
            # Determine overall status
            if critical_count > 0:
                component.overall_status = HealthStatus.CRITICAL
            elif warning_count > 0:
                component.overall_status = HealthStatus.WARNING
            elif component.health_score >= 0.8:
                component.overall_status = HealthStatus.HEALTHY
            elif component.health_score >= 0.5:
                component.overall_status = HealthStatus.DEGRADED
            else:
                component.overall_status = HealthStatus.WARNING
            
            # Store health history
            if self.enable_trend_analysis:
                self.health_history[component_id].append({
                    'timestamp': datetime.now(),
                    'health_score': component.health_score,
                    'status': component.overall_status.value
                })
            
        except Exception as e:
            logger.error(f"Error calculating component health for {component_id}: {e}")
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        logger.info("Health monitor loop started")
        
        while self.running:
            try:
                # Clean old health history
                await self._clean_health_history()
                
                # Perform trend analysis if enabled
                if self.enable_trend_analysis:
                    await self._analyze_health_trends()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("Health monitor loop stopped")
    
    async def _clean_health_history(self):
        """Clean old health history data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.health_retention_hours)
            
            for component_id, history in self.health_history.items():
                while history and history[0]['timestamp'] < cutoff_time:
                    history.popleft()
            
        except Exception as e:
            logger.error(f"Error cleaning health history: {e}")
    
    async def _analyze_health_trends(self):
        """Analyze health trends for components"""
        try:
            for component_id, history in self.health_history.items():
                if len(history) >= 10:  # Minimum data points for trend analysis
                    health_scores = [entry['health_score'] for entry in history]
                    
                    # Simple trend analysis
                    recent_scores = health_scores[-10:]
                    older_scores = health_scores[-20:-10] if len(health_scores) >= 20 else health_scores[:-10]
                    
                    if older_scores:
                        recent_avg = sum(recent_scores) / len(recent_scores)
                        older_avg = sum(older_scores) / len(older_scores)
                        
                        trend_direction = "stable"
                        trend_strength = 0.0
                        
                        if recent_avg > older_avg + 0.1:
                            trend_direction = "improving"
                            trend_strength = min((recent_avg - older_avg) / 0.5, 1.0)
                        elif recent_avg < older_avg - 0.1:
                            trend_direction = "degrading"
                            trend_strength = min((older_avg - recent_avg) / 0.5, 1.0)
                        
                        # Store trend information (could be used for predictions)
                        logger.debug(f"Health trend for {component_id}: {trend_direction} (strength: {trend_strength:.3f})")
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        try:
            # Calculate overall system health
            if not self.component_health:
                return SystemHealth(
                    overall_status=HealthStatus.UNKNOWN,
                    overall_score=0.0,
                    component_health={},
                    failed_checks=[],
                    warning_checks=[],
                    total_checks=0,
                    passed_checks=0,
                    timestamp=datetime.now()
                )
            
            # Aggregate component health
            total_score = sum(comp.health_score for comp in self.component_health.values())
            overall_score = total_score / len(self.component_health)
            
            critical_components = sum(1 for comp in self.component_health.values() 
                                    if comp.overall_status == HealthStatus.CRITICAL)
            warning_components = sum(1 for comp in self.component_health.values() 
                                   if comp.overall_status == HealthStatus.WARNING)
            
            # Determine overall status
            if critical_components > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_components > 0:
                overall_status = HealthStatus.WARNING
            elif overall_score >= 0.8:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
            
            # Collect failed and warning checks
            failed_checks = []
            warning_checks = []
            total_checks = 0
            passed_checks = 0
            
            for result in self.check_results.values():
                total_checks += 1
                if result.status == HealthStatus.CRITICAL:
                    failed_checks.append(result.check_id)
                elif result.status == HealthStatus.WARNING:
                    warning_checks.append(result.check_id)
                elif result.status == HealthStatus.HEALTHY:
                    passed_checks += 1
            
            # Health summary
            health_summary = {
                'total_components': len(self.component_health),
                'healthy_components': sum(1 for comp in self.component_health.values() 
                                        if comp.overall_status == HealthStatus.HEALTHY),
                'warning_components': warning_components,
                'critical_components': critical_components,
                'unknown_components': sum(1 for comp in self.component_health.values() 
                                        if comp.overall_status == HealthStatus.UNKNOWN),
                'average_health_score': overall_score
            }
            
            return SystemHealth(
                overall_status=overall_status,
                overall_score=overall_score,
                component_health=self.component_health.copy(),
                failed_checks=failed_checks,
                warning_checks=warning_checks,
                total_checks=total_checks,
                passed_checks=passed_checks,
                timestamp=datetime.now(),
                health_summary=health_summary
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                overall_score=0.0,
                component_health={},
                failed_checks=[],
                warning_checks=[],
                total_checks=0,
                passed_checks=0,
                timestamp=datetime.now()
            )
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status for specific component"""
        return self.component_health.get(component_id)
    
    def get_check_result(self, check_id: str) -> Optional[HealthCheckResult]:
        """Get result of specific health check"""
        return self.check_results.get(check_id)
    
    def get_health_history(self, component_id: str, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for component"""
        if component_id not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        return [entry for entry in self.health_history[component_id] 
                if entry['timestamp'] >= cutoff_time]
    
    def get_status(self) -> Dict[str, Any]:
        """Get health checker status"""
        system_health = self.get_system_health()
        
        return {
            'running': self.running,
            'registered_checks': len(self.health_checks),
            'active_check_tasks': len(self.check_tasks),
            'monitored_components': len(self.component_health),
            'overall_system_status': system_health.overall_status.value,
            'overall_system_score': system_health.overall_score,
            'total_checks_executed': len(self.check_results),
            'failed_checks': len(system_health.failed_checks),
            'warning_checks': len(system_health.warning_checks),
            'passed_checks': system_health.passed_checks,
            'enable_trend_analysis': self.enable_trend_analysis,
            'parallel_checks': self.parallel_checks
        }

# Test and demonstration
async def test_health_checker():
    """Test the health checker system"""
    print("🧠 Testing CortexOS Health Checker...")
    
    # Create configuration
    config = {
        'default_check_interval': 10,  # 10 seconds for testing
        'enable_trend_analysis': True,
        'health_retention_hours': 1,   # 1 hour for testing
        'parallel_checks': True,
        'max_concurrent_checks': 5
    }
    
    # Initialize health checker
    checker = HealthChecker(config)
    
    # Register test component
    def test_component_ping():
        return True
    
    def test_component_memory():
        return 150.0  # MB
    
    def test_component_errors():
        return 2, 100  # 2 errors out of 100 operations
    
    custom_checks = [
        HealthCheck(
            check_id="responsiveness",
            name="Component Responsiveness",
            description="Check if component responds to ping",
            check_type=CheckType.COMPONENT,
            severity=CheckSeverity.HIGH,
            check_function=lambda: ComponentHealthChecks.check_component_responsiveness("test_component", test_component_ping),
            interval=15
        ),
        HealthCheck(
            check_id="memory_usage",
            name="Component Memory Usage",
            description="Monitor component memory consumption",
            check_type=CheckType.COMPONENT,
            severity=CheckSeverity.MEDIUM,
            check_function=lambda: ComponentHealthChecks.check_component_memory_usage("test_component", test_component_memory),
            interval=30
        ),
        HealthCheck(
            check_id="error_rate",
            name="Component Error Rate",
            description="Monitor component error rate",
            check_type=CheckType.COMPONENT,
            severity=CheckSeverity.HIGH,
            check_function=lambda: ComponentHealthChecks.check_component_error_rate("test_component", test_component_errors),
            interval=60
        )
    ]
    
    checker.register_component("test_component", "Test Component", custom_checks)
    
    try:
        # Start health checking
        await checker.start()
        print("✅ Health Checker started")
        
        # Wait for health checks to run
        print("\n⏳ Running health checks...")
        await asyncio.sleep(20)
        
        # Get system health
        system_health = checker.get_system_health()
        print(f"\n🏥 System Health Report:")
        print(f"   Overall Status: {system_health.overall_status.value}")
        print(f"   Overall Score: {system_health.overall_score:.3f}")
        print(f"   Total Checks: {system_health.total_checks}")
        print(f"   Passed Checks: {system_health.passed_checks}")
        print(f"   Failed Checks: {len(system_health.failed_checks)}")
        print(f"   Warning Checks: {len(system_health.warning_checks)}")
        
        # Display component health
        print(f"\n🔧 Component Health:")
        for comp_id, comp_health in system_health.component_health.items():
            print(f"   {comp_health.component_name}: {comp_health.overall_status.value} (Score: {comp_health.health_score:.3f})")
            print(f"      Checks: {len(comp_health.check_results)}, Errors: {comp_health.error_count}, Warnings: {comp_health.warning_count}")
        
        # Display individual check results
        print(f"\n📋 Recent Check Results:")
        for check_id, result in checker.check_results.items():
            status_emoji = "✅" if result.status == HealthStatus.HEALTHY else "⚠️" if result.status == HealthStatus.WARNING else "❌"
            print(f"   {status_emoji} {check_id}: {result.message}")
            print(f"      Execution time: {result.execution_time:.3f}s")
        
        # Display health summary
        print(f"\n📊 Health Summary:")
        summary = system_health.health_summary
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Display checker status
        print(f"\n🔧 Checker Status:")
        status = checker.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Health Checker test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await checker.stop()

if __name__ == "__main__":
    asyncio.run(test_health_checker())

