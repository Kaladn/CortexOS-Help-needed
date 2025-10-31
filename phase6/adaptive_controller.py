#!/usr/bin/env python3
"""
CortexOS Phase 6: Adaptive Controller
Master adaptive control system for coordinating all CortexOS components
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ControlMode(Enum):
    """Adaptive control modes"""
    MANUAL = "manual"
    SEMI_AUTOMATIC = "semi_automatic"
    AUTOMATIC = "automatic"
    AUTONOMOUS = "autonomous"
    EMERGENCY = "emergency"

class AdaptationStrategy(Enum):
    """Adaptation strategies"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    LEARNING = "learning"
    HYBRID = "hybrid"

class SystemState(Enum):
    """Overall system states"""
    INITIALIZING = "initializing"
    STABLE = "stable"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"

class ComponentType(Enum):
    """CortexOS component types"""
    INFRASTRUCTURE = "infrastructure"
    NEURAL_ENGINE = "neural_engine"
    CONTEXT_ENGINE = "context_engine"
    GATEKEEPER = "gatekeeper"
    RESONANCE_FIELD = "resonance_field"
    RESONANCE_MONITOR = "resonance_monitor"
    RESONANCE_REINFORCER = "resonance_reinforcer"
    TOPK_SPARSE = "topk_sparse"
    MEMORY_INSERTER = "memory_inserter"
    MEMORY_RETRIEVER = "memory_retriever"
    MEMORY_CONSOLIDATOR = "memory_consolidator"
    COGNITIVE_BRIDGE = "cognitive_bridge"
    DATA_INGESTION = "data_ingestion"
    STREAM_PROCESSOR = "stream_processor"
    BATCH_PROCESSOR = "batch_processor"
    INGESTION_VALIDATOR = "ingestion_validator"
    SYSTEM_MONITOR = "system_monitor"
    PERFORMANCE_TRACKER = "performance_tracker"
    HEALTH_CHECKER = "health_checker"
    ALERT_MANAGER = "alert_manager"
    MOOD_MODULATOR = "mood_modulator"
    COGNITIVE_ENHANCER = "cognitive_enhancer"
    NEURAL_OPTIMIZER = "neural_optimizer"

@dataclass
class ComponentStatus:
    """Status of individual component"""
    component_id: str
    component_type: ComponentType
    status: str  # running, stopped, error, degraded
    health_score: float  # 0.0 to 1.0
    performance_metrics: Dict[str, float]
    last_update: datetime
    error_count: int = 0
    warning_count: int = 0
    uptime: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class AdaptationRequest:
    """Request for system adaptation"""
    request_id: str
    trigger_type: str  # performance, error, user, scheduled
    target_components: List[ComponentType]
    adaptation_type: str  # configuration, resource, behavior
    parameters: Dict[str, Any]
    priority: int = 1  # 1-10
    timeout_seconds: int = 300
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptationResult:
    """Result of adaptation"""
    result_id: str
    request_id: str
    success: bool
    components_affected: List[ComponentType]
    changes_made: Dict[str, Any]
    performance_impact: Dict[str, float]
    side_effects: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    overall_health: float  # 0.0 to 1.0
    performance_score: float  # 0.0 to 1.0
    stability_index: float  # 0.0 to 1.0
    adaptation_rate: float  # adaptations per hour
    error_rate: float  # errors per hour
    resource_efficiency: float  # 0.0 to 1.0
    cognitive_load: float  # 0.0 to 1.0
    response_time: float  # milliseconds
    throughput: float  # operations per second
    energy_consumption: float  # watts
    timestamp: datetime = field(default_factory=datetime.now)

class ComponentManager:
    """Manager for CortexOS components"""
    
    def __init__(self):
        self.components = {}
        self.component_dependencies = self._initialize_dependencies()
        self.component_interfaces = {}
        self.startup_order = self._initialize_startup_order()
    
    def _initialize_dependencies(self) -> Dict[ComponentType, List[ComponentType]]:
        """Initialize component dependencies"""
        dependencies = {}
        
        # Infrastructure components (no dependencies)
        dependencies[ComponentType.INFRASTRUCTURE] = []
        
        # Phase 1 components depend on infrastructure
        dependencies[ComponentType.NEURAL_ENGINE] = [ComponentType.INFRASTRUCTURE]
        dependencies[ComponentType.CONTEXT_ENGINE] = [ComponentType.INFRASTRUCTURE]
        dependencies[ComponentType.GATEKEEPER] = [ComponentType.INFRASTRUCTURE]
        
        # Phase 2 components depend on Phase 1
        dependencies[ComponentType.RESONANCE_FIELD] = [ComponentType.NEURAL_ENGINE, ComponentType.CONTEXT_ENGINE]
        dependencies[ComponentType.RESONANCE_MONITOR] = [ComponentType.RESONANCE_FIELD]
        dependencies[ComponentType.RESONANCE_REINFORCER] = [ComponentType.RESONANCE_FIELD]
        dependencies[ComponentType.TOPK_SPARSE] = [ComponentType.RESONANCE_FIELD]
        
        # Phase 3 components depend on Phase 2
        dependencies[ComponentType.MEMORY_INSERTER] = [ComponentType.RESONANCE_FIELD]
        dependencies[ComponentType.MEMORY_RETRIEVER] = [ComponentType.MEMORY_INSERTER]
        dependencies[ComponentType.MEMORY_CONSOLIDATOR] = [ComponentType.MEMORY_INSERTER]
        dependencies[ComponentType.COGNITIVE_BRIDGE] = [ComponentType.MEMORY_INSERTER, ComponentType.NEURAL_ENGINE]
        
        # Phase 4 components depend on Phase 3
        dependencies[ComponentType.DATA_INGESTION] = [ComponentType.COGNITIVE_BRIDGE]
        dependencies[ComponentType.STREAM_PROCESSOR] = [ComponentType.DATA_INGESTION]
        dependencies[ComponentType.BATCH_PROCESSOR] = [ComponentType.DATA_INGESTION]
        dependencies[ComponentType.INGESTION_VALIDATOR] = [ComponentType.DATA_INGESTION]
        
        # Phase 5 components depend on all previous phases
        dependencies[ComponentType.SYSTEM_MONITOR] = [ComponentType.INFRASTRUCTURE]
        dependencies[ComponentType.PERFORMANCE_TRACKER] = [ComponentType.SYSTEM_MONITOR]
        dependencies[ComponentType.HEALTH_CHECKER] = [ComponentType.SYSTEM_MONITOR]
        dependencies[ComponentType.ALERT_MANAGER] = [ComponentType.HEALTH_CHECKER]
        
        # Phase 6 components depend on monitoring
        dependencies[ComponentType.MOOD_MODULATOR] = [ComponentType.HEALTH_CHECKER]
        dependencies[ComponentType.COGNITIVE_ENHANCER] = [ComponentType.PERFORMANCE_TRACKER]
        dependencies[ComponentType.NEURAL_OPTIMIZER] = [ComponentType.PERFORMANCE_TRACKER]
        
        return dependencies
    
    def _initialize_startup_order(self) -> List[ComponentType]:
        """Initialize component startup order based on dependencies"""
        return [
            # Infrastructure first
            ComponentType.INFRASTRUCTURE,
            
            # Phase 1: Core engines
            ComponentType.NEURAL_ENGINE,
            ComponentType.CONTEXT_ENGINE,
            ComponentType.GATEKEEPER,
            
            # Phase 2: Resonance system
            ComponentType.RESONANCE_FIELD,
            ComponentType.RESONANCE_MONITOR,
            ComponentType.RESONANCE_REINFORCER,
            ComponentType.TOPK_SPARSE,
            
            # Phase 3: Memory system
            ComponentType.MEMORY_INSERTER,
            ComponentType.MEMORY_RETRIEVER,
            ComponentType.MEMORY_CONSOLIDATOR,
            ComponentType.COGNITIVE_BRIDGE,
            
            # Phase 4: Ingestion pipeline
            ComponentType.DATA_INGESTION,
            ComponentType.STREAM_PROCESSOR,
            ComponentType.BATCH_PROCESSOR,
            ComponentType.INGESTION_VALIDATOR,
            
            # Phase 5: Monitoring system
            ComponentType.SYSTEM_MONITOR,
            ComponentType.PERFORMANCE_TRACKER,
            ComponentType.HEALTH_CHECKER,
            ComponentType.ALERT_MANAGER,
            
            # Phase 6: Modulation system
            ComponentType.MOOD_MODULATOR,
            ComponentType.COGNITIVE_ENHANCER,
            ComponentType.NEURAL_OPTIMIZER
        ]
    
    def register_component(self, component_type: ComponentType, component_interface: Any):
        """Register component with manager"""
        self.component_interfaces[component_type] = component_interface
        
        # Initialize component status
        self.components[component_type] = ComponentStatus(
            component_id=f"{component_type.value}_{int(time.time())}",
            component_type=component_type,
            status="registered",
            health_score=1.0,
            performance_metrics={},
            last_update=datetime.now()
        )
        
        logger.info(f"Component registered: {component_type.value}")
    
    async def start_component(self, component_type: ComponentType) -> bool:
        """Start individual component"""
        try:
            if component_type not in self.component_interfaces:
                logger.error(f"Component not registered: {component_type.value}")
                return False
            
            # Check dependencies
            dependencies = self.component_dependencies.get(component_type, [])
            for dep in dependencies:
                if dep not in self.components or self.components[dep].status != "running":
                    logger.error(f"Dependency not running: {dep.value} for {component_type.value}")
                    return False
            
            # Start component
            component_interface = self.component_interfaces[component_type]
            if hasattr(component_interface, 'start'):
                await component_interface.start()
            
            # Update status
            self.components[component_type].status = "running"
            self.components[component_type].last_update = datetime.now()
            
            logger.info(f"Component started: {component_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting component {component_type.value}: {e}")
            if component_type in self.components:
                self.components[component_type].status = "error"
                self.components[component_type].error_count += 1
            return False
    
    async def stop_component(self, component_type: ComponentType) -> bool:
        """Stop individual component"""
        try:
            if component_type not in self.component_interfaces:
                return False
            
            # Stop component
            component_interface = self.component_interfaces[component_type]
            if hasattr(component_interface, 'stop'):
                await component_interface.stop()
            
            # Update status
            self.components[component_type].status = "stopped"
            self.components[component_type].last_update = datetime.now()
            
            logger.info(f"Component stopped: {component_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping component {component_type.value}: {e}")
            return False
    
    async def restart_component(self, component_type: ComponentType) -> bool:
        """Restart component"""
        success = await self.stop_component(component_type)
        if success:
            await asyncio.sleep(1)  # Brief pause
            success = await self.start_component(component_type)
        return success
    
    def get_component_status(self, component_type: ComponentType) -> Optional[ComponentStatus]:
        """Get component status"""
        return self.components.get(component_type)
    
    def get_all_components_status(self) -> Dict[ComponentType, ComponentStatus]:
        """Get status of all components"""
        return self.components.copy()
    
    def update_component_metrics(self, component_type: ComponentType, metrics: Dict[str, float]):
        """Update component performance metrics"""
        if component_type in self.components:
            self.components[component_type].performance_metrics.update(metrics)
            self.components[component_type].last_update = datetime.now()

class AdaptationEngine:
    """Adaptive control engine"""
    
    def __init__(self):
        self.adaptation_strategies = {
            AdaptationStrategy.REACTIVE: self._reactive_adaptation,
            AdaptationStrategy.PROACTIVE: self._proactive_adaptation,
            AdaptationStrategy.PREDICTIVE: self._predictive_adaptation,
            AdaptationStrategy.LEARNING: self._learning_adaptation,
            AdaptationStrategy.HYBRID: self._hybrid_adaptation
        }
        
        self.adaptation_history = deque(maxlen=1000)
        self.performance_baselines = {}
        self.learning_models = {}
    
    async def execute_adaptation(self, request: AdaptationRequest, 
                                component_manager: ComponentManager,
                                strategy: AdaptationStrategy) -> AdaptationResult:
        """Execute adaptation request"""
        try:
            start_time = time.time()
            
            # Get adaptation strategy
            strategy_func = self.adaptation_strategies.get(strategy, self._reactive_adaptation)
            
            # Execute adaptation
            result = await strategy_func(request, component_manager)
            
            # Calculate execution time
            result.execution_time = time.time() - start_time
            
            # Store in history
            self.adaptation_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing adaptation: {e}")
            return AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=False,
                components_affected=[],
                changes_made={},
                performance_impact={},
                side_effects=[f"Adaptation failed: {e}"]
            )
    
    async def _reactive_adaptation(self, request: AdaptationRequest, 
                                 component_manager: ComponentManager) -> AdaptationResult:
        """Reactive adaptation - respond to current issues"""
        changes_made = {}
        components_affected = []
        performance_impact = {}
        
        try:
            for component_type in request.target_components:
                component_status = component_manager.get_component_status(component_type)
                
                if not component_status:
                    continue
                
                # React to component issues
                if component_status.health_score < 0.5:
                    # Restart unhealthy component
                    success = await component_manager.restart_component(component_type)
                    if success:
                        changes_made[component_type.value] = "restarted"
                        components_affected.append(component_type)
                        performance_impact[component_type.value] = 0.2  # Expected improvement
                
                elif component_status.error_count > 10:
                    # Reset error count and apply conservative settings
                    component_status.error_count = 0
                    changes_made[component_type.value] = "error_reset"
                    components_affected.append(component_type)
            
            result = AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=len(components_affected) > 0,
                components_affected=components_affected,
                changes_made=changes_made,
                performance_impact=performance_impact
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reactive adaptation: {e}")
            raise
    
    async def _proactive_adaptation(self, request: AdaptationRequest,
                                  component_manager: ComponentManager) -> AdaptationResult:
        """Proactive adaptation - prevent issues before they occur"""
        changes_made = {}
        components_affected = []
        performance_impact = {}
        
        try:
            for component_type in request.target_components:
                component_status = component_manager.get_component_status(component_type)
                
                if not component_status:
                    continue
                
                # Proactive measures
                if component_status.health_score < 0.8:
                    # Preemptive optimization
                    changes_made[component_type.value] = "preemptive_optimization"
                    components_affected.append(component_type)
                    performance_impact[component_type.value] = 0.1
                
                # Resource reallocation
                if 'cpu_usage' in component_status.resource_usage:
                    if component_status.resource_usage['cpu_usage'] > 0.8:
                        changes_made[component_type.value] = "resource_reallocation"
                        components_affected.append(component_type)
                        performance_impact[component_type.value] = 0.15
            
            result = AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=len(components_affected) > 0,
                components_affected=components_affected,
                changes_made=changes_made,
                performance_impact=performance_impact
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in proactive adaptation: {e}")
            raise
    
    async def _predictive_adaptation(self, request: AdaptationRequest,
                                   component_manager: ComponentManager) -> AdaptationResult:
        """Predictive adaptation - adapt based on predicted future needs"""
        changes_made = {}
        components_affected = []
        performance_impact = {}
        
        try:
            # Analyze trends and predict future needs
            for component_type in request.target_components:
                component_status = component_manager.get_component_status(component_type)
                
                if not component_status:
                    continue
                
                # Predict future performance based on trends
                predicted_health = self._predict_component_health(component_type, component_status)
                
                if predicted_health < 0.6:
                    # Preemptive scaling or optimization
                    changes_made[component_type.value] = "predictive_scaling"
                    components_affected.append(component_type)
                    performance_impact[component_type.value] = 0.25
            
            result = AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=len(components_affected) > 0,
                components_affected=components_affected,
                changes_made=changes_made,
                performance_impact=performance_impact
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predictive adaptation: {e}")
            raise
    
    async def _learning_adaptation(self, request: AdaptationRequest,
                                 component_manager: ComponentManager) -> AdaptationResult:
        """Learning adaptation - adapt based on learned patterns"""
        changes_made = {}
        components_affected = []
        performance_impact = {}
        
        try:
            # Learn from adaptation history
            successful_adaptations = [a for a in self.adaptation_history if a.success]
            
            for component_type in request.target_components:
                # Find similar past adaptations
                similar_adaptations = [
                    a for a in successful_adaptations 
                    if component_type in a.components_affected
                ]
                
                if similar_adaptations:
                    # Apply learned adaptation pattern
                    best_adaptation = max(similar_adaptations, 
                                        key=lambda a: sum(a.performance_impact.values()))
                    
                    # Replicate successful pattern
                    for change_type in best_adaptation.changes_made.values():
                        changes_made[component_type.value] = f"learned_{change_type}"
                        components_affected.append(component_type)
                        performance_impact[component_type.value] = 0.2
            
            result = AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=len(components_affected) > 0,
                components_affected=components_affected,
                changes_made=changes_made,
                performance_impact=performance_impact
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in learning adaptation: {e}")
            raise
    
    async def _hybrid_adaptation(self, request: AdaptationRequest,
                               component_manager: ComponentManager) -> AdaptationResult:
        """Hybrid adaptation - combine multiple strategies"""
        try:
            # Execute multiple strategies and combine results
            reactive_result = await self._reactive_adaptation(request, component_manager)
            proactive_result = await self._proactive_adaptation(request, component_manager)
            
            # Combine results
            combined_changes = {}
            combined_changes.update(reactive_result.changes_made)
            combined_changes.update(proactive_result.changes_made)
            
            combined_components = list(set(reactive_result.components_affected + proactive_result.components_affected))
            
            combined_impact = {}
            combined_impact.update(reactive_result.performance_impact)
            for comp, impact in proactive_result.performance_impact.items():
                combined_impact[comp] = combined_impact.get(comp, 0) + impact
            
            result = AdaptationResult(
                result_id=f"result_{request.request_id}",
                request_id=request.request_id,
                success=len(combined_components) > 0,
                components_affected=combined_components,
                changes_made=combined_changes,
                performance_impact=combined_impact
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid adaptation: {e}")
            raise
    
    def _predict_component_health(self, component_type: ComponentType, 
                                component_status: ComponentStatus) -> float:
        """Predict future component health"""
        # Simplified prediction based on current trends
        current_health = component_status.health_score
        error_trend = min(component_status.error_count / 100.0, 0.5)  # Cap at 0.5
        
        # Simple linear prediction
        predicted_health = current_health - error_trend * 0.1
        
        return max(0.0, min(1.0, predicted_health))

class AdaptiveController:
    """Master adaptive control system for coordinating all CortexOS components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.component_manager = ComponentManager()
        self.adaptation_engine = AdaptationEngine()
        
        # Current state
        self.system_state = SystemState.INITIALIZING
        self.control_mode = ControlMode.AUTOMATIC
        self.adaptation_strategy = AdaptationStrategy.HYBRID
        
        # Queues and tracking
        self.adaptation_queue = asyncio.Queue()
        self.active_adaptations = {}
        self.system_metrics_history = deque(maxlen=1000)
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        self.adaptation_interval = self.config.get('adaptation_interval', 60)  # seconds
        self.health_check_interval = self.config.get('health_check_interval', 10)  # seconds
        self.max_concurrent_adaptations = self.config.get('max_concurrent_adaptations', 3)
        self.enable_auto_adaptation = self.config.get('enable_auto_adaptation', True)
        
        # Thresholds
        self.health_threshold = self.config.get('health_threshold', 0.7)
        self.performance_threshold = self.config.get('performance_threshold', 0.6)
        self.stability_threshold = self.config.get('stability_threshold', 0.8)
        
        # State
        self.running = False
        self.monitoring_task = None
        self.adaptation_task = None
        self.health_check_task = None
        self.auto_adaptation_task = None
        
        logger.info("Adaptive Controller initialized")
    
    async def start(self):
        """Start adaptive controller"""
        try:
            self.running = True
            self.system_state = SystemState.INITIALIZING
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start adaptation processing task
            self.adaptation_task = asyncio.create_task(self._adaptation_loop())
            
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start auto-adaptation task if enabled
            if self.enable_auto_adaptation:
                self.auto_adaptation_task = asyncio.create_task(self._auto_adaptation_loop())
            
            # Initialize system
            await self._initialize_system()
            
            self.system_state = SystemState.STABLE
            logger.info("Adaptive Controller started")
            
        except Exception as e:
            logger.error(f"Error starting Adaptive Controller: {e}")
            self.system_state = SystemState.CRITICAL
            raise
    
    async def stop(self):
        """Stop adaptive controller"""
        try:
            self.running = False
            self.system_state = SystemState.SHUTDOWN
            
            # Stop all components in reverse order
            await self._shutdown_system()
            
            # Cancel tasks
            tasks = [self.monitoring_task, self.adaptation_task, 
                    self.health_check_task, self.auto_adaptation_task]
            for task in tasks:
                if task:
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
            
            logger.info("Adaptive Controller stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Adaptive Controller: {e}")
    
    async def _initialize_system(self):
        """Initialize CortexOS system"""
        try:
            logger.info("Initializing CortexOS system...")
            
            # Start components in dependency order
            startup_order = self.component_manager.startup_order
            
            for component_type in startup_order:
                if component_type in self.component_manager.component_interfaces:
                    success = await self.component_manager.start_component(component_type)
                    if not success:
                        logger.warning(f"Failed to start component: {component_type.value}")
                    else:
                        logger.info(f"Started component: {component_type.value}")
                    
                    # Brief pause between component starts
                    await asyncio.sleep(0.5)
            
            logger.info("CortexOS system initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    async def _shutdown_system(self):
        """Shutdown CortexOS system"""
        try:
            logger.info("Shutting down CortexOS system...")
            
            # Stop components in reverse order
            shutdown_order = list(reversed(self.component_manager.startup_order))
            
            for component_type in shutdown_order:
                if component_type in self.component_manager.component_interfaces:
                    success = await self.component_manager.stop_component(component_type)
                    if success:
                        logger.info(f"Stopped component: {component_type.value}")
                    
                    # Brief pause between component stops
                    await asyncio.sleep(0.2)
            
            logger.info("CortexOS system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error shutting down system: {e}")
    
    async def request_adaptation(self, request: AdaptationRequest) -> str:
        """Request system adaptation"""
        try:
            # Validate request
            if not self._validate_adaptation_request(request):
                raise ValueError("Invalid adaptation request")
            
            # Add to queue
            await self.adaptation_queue.put(request)
            
            logger.info(f"Adaptation requested: {request.request_id}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Error requesting adaptation: {e}")
            raise
    
    def _validate_adaptation_request(self, request: AdaptationRequest) -> bool:
        """Validate adaptation request"""
        try:
            # Check concurrent adaptations
            if len(self.active_adaptations) >= self.max_concurrent_adaptations:
                logger.warning("Too many concurrent adaptations")
                return False
            
            # Check target components
            if not request.target_components:
                logger.warning("No target components specified")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating adaptation request: {e}")
            return False
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update system state
                await self._update_system_state()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Monitoring loop stopped")
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        try:
            components_status = self.component_manager.get_all_components_status()
            
            # Calculate overall metrics
            running_components = [c for c in components_status.values() if c.status == "running"]
            
            if not running_components:
                overall_health = 0.0
                performance_score = 0.0
                stability_index = 0.0
            else:
                overall_health = sum(c.health_score for c in running_components) / len(running_components)
                
                # Performance score based on component metrics
                performance_scores = []
                for component in running_components:
                    if component.performance_metrics:
                        avg_performance = sum(component.performance_metrics.values()) / len(component.performance_metrics)
                        performance_scores.append(avg_performance)
                
                performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0.5
                
                # Stability based on error rates
                total_errors = sum(c.error_count for c in running_components)
                stability_index = max(0.0, 1.0 - (total_errors / (len(running_components) * 100)))
            
            # Calculate other metrics
            adaptation_rate = len(self.adaptation_engine.adaptation_history) / max(1, len(self.system_metrics_history))
            error_rate = sum(c.error_count for c in running_components) / max(1, len(running_components))
            
            # Resource efficiency (simplified)
            resource_efficiency = overall_health * performance_score
            
            # Cognitive load (based on active adaptations and system complexity)
            cognitive_load = min(1.0, len(self.active_adaptations) / 10.0 + (1.0 - stability_index) * 0.5)
            
            # Response time and throughput (simulated)
            response_time = 100 + (1.0 - performance_score) * 200  # ms
            throughput = performance_score * 1000  # ops/sec
            
            # Energy consumption (simulated)
            energy_consumption = 100 + len(running_components) * 20  # watts
            
            metrics = SystemMetrics(
                overall_health=overall_health,
                performance_score=performance_score,
                stability_index=stability_index,
                adaptation_rate=adaptation_rate,
                error_rate=error_rate,
                resource_efficiency=resource_efficiency,
                cognitive_load=cognitive_load,
                response_time=response_time,
                throughput=throughput,
                energy_consumption=energy_consumption
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _update_system_state(self):
        """Update overall system state"""
        try:
            if not self.system_metrics_history:
                return
            
            current_metrics = self.system_metrics_history[-1]
            
            # Determine system state based on metrics
            if current_metrics.overall_health < 0.3:
                self.system_state = SystemState.CRITICAL
            elif current_metrics.overall_health < self.health_threshold:
                self.system_state = SystemState.DEGRADED
            elif len(self.active_adaptations) > 0:
                self.system_state = SystemState.ADAPTING
            elif current_metrics.performance_score < self.performance_threshold:
                self.system_state = SystemState.OPTIMIZING
            else:
                self.system_state = SystemState.STABLE
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    async def _adaptation_loop(self):
        """Adaptation processing loop"""
        logger.info("Adaptation loop started")
        
        while self.running:
            try:
                # Process adaptation requests
                try:
                    request = await asyncio.wait_for(self.adaptation_queue.get(), timeout=1.0)
                    await self._process_adaptation_request(request)
                except asyncio.TimeoutError:
                    pass
                
                # Update active adaptations
                await self._update_active_adaptations()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(1)
        
        logger.info("Adaptation loop stopped")
    
    async def _process_adaptation_request(self, request: AdaptationRequest):
        """Process individual adaptation request"""
        try:
            # Start adaptation
            self.active_adaptations[request.request_id] = {
                'request': request,
                'start_time': time.time(),
                'status': 'running'
            }
            
            # Execute adaptation
            result = await self.adaptation_engine.execute_adaptation(
                request, self.component_manager, self.adaptation_strategy
            )
            
            # Update active adaptation
            if request.request_id in self.active_adaptations:
                self.active_adaptations[request.request_id]['status'] = 'completed'
                self.active_adaptations[request.request_id]['result'] = result
            
            logger.info(f"Adaptation completed: {request.request_id}")
            
        except Exception as e:
            logger.error(f"Error processing adaptation request: {e}")
            if request.request_id in self.active_adaptations:
                self.active_adaptations[request.request_id]['status'] = 'failed'
    
    async def _update_active_adaptations(self):
        """Update and clean up active adaptations"""
        try:
            completed_adaptations = []
            
            for request_id, adaptation in self.active_adaptations.items():
                # Remove completed adaptations after some time
                if adaptation['status'] in ['completed', 'failed']:
                    elapsed_time = time.time() - adaptation['start_time']
                    if elapsed_time > 300:  # 5 minutes
                        completed_adaptations.append(request_id)
            
            for request_id in completed_adaptations:
                del self.active_adaptations[request_id]
            
        except Exception as e:
            logger.error(f"Error updating active adaptations: {e}")
    
    async def _health_check_loop(self):
        """Health check loop"""
        logger.info("Health check loop started")
        
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Health check loop stopped")
    
    async def _perform_health_checks(self):
        """Perform system health checks"""
        try:
            components_status = self.component_manager.get_all_components_status()
            
            for component_type, status in components_status.items():
                # Update component health based on various factors
                health_score = 1.0
                
                # Reduce health based on errors
                if status.error_count > 0:
                    health_score -= min(0.5, status.error_count / 100.0)
                
                # Reduce health based on uptime issues
                if status.status != "running":
                    health_score -= 0.3
                
                # Update health score
                status.health_score = max(0.0, health_score)
                status.last_update = datetime.now()
                
                # Update component metrics
                self.component_manager.update_component_metrics(component_type, {
                    'health_score': health_score,
                    'uptime': time.time() - status.last_update.timestamp()
                })
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _auto_adaptation_loop(self):
        """Automatic adaptation loop"""
        logger.info("Auto-adaptation loop started")
        
        while self.running:
            try:
                await self._perform_auto_adaptation()
                await asyncio.sleep(self.adaptation_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-adaptation loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("Auto-adaptation loop stopped")
    
    async def _perform_auto_adaptation(self):
        """Perform automatic adaptation based on system state"""
        try:
            if not self.system_metrics_history or len(self.active_adaptations) >= 2:
                return
            
            current_metrics = self.system_metrics_history[-1]
            
            # Determine if adaptation is needed
            adaptation_needed = False
            target_components = []
            
            # Check overall health
            if current_metrics.overall_health < self.health_threshold:
                adaptation_needed = True
                # Target all components with low health
                components_status = self.component_manager.get_all_components_status()
                for comp_type, status in components_status.items():
                    if status.health_score < self.health_threshold:
                        target_components.append(comp_type)
            
            # Check performance
            elif current_metrics.performance_score < self.performance_threshold:
                adaptation_needed = True
                # Target performance-critical components
                target_components = [
                    ComponentType.NEURAL_ENGINE,
                    ComponentType.COGNITIVE_ENHANCER,
                    ComponentType.NEURAL_OPTIMIZER
                ]
            
            # Check stability
            elif current_metrics.stability_index < self.stability_threshold:
                adaptation_needed = True
                # Target monitoring and control components
                target_components = [
                    ComponentType.SYSTEM_MONITOR,
                    ComponentType.HEALTH_CHECKER,
                    ComponentType.ALERT_MANAGER
                ]
            
            if adaptation_needed and target_components:
                # Create auto-adaptation request
                auto_request = AdaptationRequest(
                    request_id=f"auto_adapt_{int(time.time())}",
                    trigger_type="automatic",
                    target_components=target_components[:3],  # Limit to 3 components
                    adaptation_type="optimization",
                    parameters={
                        'health_threshold': self.health_threshold,
                        'performance_threshold': self.performance_threshold
                    },
                    context={'auto_adaptation': True}
                )
                
                await self.request_adaptation(auto_request)
                logger.info(f"Auto-adaptation triggered for {len(target_components)} components")
            
        except Exception as e:
            logger.error(f"Error in auto-adaptation: {e}")
    
    def set_control_mode(self, mode: ControlMode):
        """Set control mode"""
        self.control_mode = mode
        logger.info(f"Control mode set to: {mode.value}")
    
    def set_adaptation_strategy(self, strategy: AdaptationStrategy):
        """Set adaptation strategy"""
        self.adaptation_strategy = strategy
        logger.info(f"Adaptation strategy set to: {strategy.value}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        components_status = self.component_manager.get_all_components_status()
        
        return {
            'system_state': self.system_state.value,
            'control_mode': self.control_mode.value,
            'adaptation_strategy': self.adaptation_strategy.value,
            'running': self.running,
            'current_metrics': {
                'overall_health': current_metrics.overall_health if current_metrics else 0.0,
                'performance_score': current_metrics.performance_score if current_metrics else 0.0,
                'stability_index': current_metrics.stability_index if current_metrics else 0.0,
                'cognitive_load': current_metrics.cognitive_load if current_metrics else 0.0,
                'response_time': current_metrics.response_time if current_metrics else 0.0,
                'throughput': current_metrics.throughput if current_metrics else 0.0
            },
            'components': {
                'total': len(components_status),
                'running': len([c for c in components_status.values() if c.status == "running"]),
                'stopped': len([c for c in components_status.values() if c.status == "stopped"]),
                'error': len([c for c in components_status.values() if c.status == "error"])
            },
            'adaptations': {
                'queue_size': self.adaptation_queue.qsize(),
                'active': len(self.active_adaptations),
                'total_completed': len(self.adaptation_engine.adaptation_history)
            },
            'enable_auto_adaptation': self.enable_auto_adaptation,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_component_details(self) -> Dict[str, Any]:
        """Get detailed component information"""
        components_status = self.component_manager.get_all_components_status()
        
        details = {}
        for comp_type, status in components_status.items():
            details[comp_type.value] = {
                'component_id': status.component_id,
                'status': status.status,
                'health_score': status.health_score,
                'error_count': status.error_count,
                'warning_count': status.warning_count,
                'uptime': status.uptime,
                'last_update': status.last_update.isoformat(),
                'performance_metrics': status.performance_metrics,
                'resource_usage': status.resource_usage
            }
        
        return details
    
    def get_adaptation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for result in self.adaptation_engine.adaptation_history:
            if result.timestamp >= cutoff_time:
                history.append({
                    'result_id': result.result_id,
                    'request_id': result.request_id,
                    'success': result.success,
                    'components_affected': [c.value for c in result.components_affected],
                    'changes_made': result.changes_made,
                    'performance_impact': {k: v for k, v in result.performance_impact.items()},
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat()
                })
        
        return history
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system analytics"""
        if not self.system_metrics_history:
            return {}
        
        recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 measurements
        
        analytics = {
            'average_health': sum(m.overall_health for m in recent_metrics) / len(recent_metrics),
            'average_performance': sum(m.performance_score for m in recent_metrics) / len(recent_metrics),
            'average_stability': sum(m.stability_index for m in recent_metrics) / len(recent_metrics),
            'average_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'average_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'total_adaptations': len(self.adaptation_engine.adaptation_history),
            'successful_adaptations': len([a for a in self.adaptation_engine.adaptation_history if a.success]),
            'adaptation_success_rate': 0.0,
            'system_uptime': time.time() if self.running else 0,
            'state_distribution': {},
            'component_health_distribution': {}
        }
        
        # Calculate adaptation success rate
        if self.adaptation_engine.adaptation_history:
            successful = len([a for a in self.adaptation_engine.adaptation_history if a.success])
            analytics['adaptation_success_rate'] = (successful / len(self.adaptation_engine.adaptation_history)) * 100
        
        # Component health distribution
        components_status = self.component_manager.get_all_components_status()
        health_ranges = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for status in components_status.values():
            if status.health_score >= 0.9:
                health_ranges['excellent'] += 1
            elif status.health_score >= 0.7:
                health_ranges['good'] += 1
            elif status.health_score >= 0.5:
                health_ranges['fair'] += 1
            else:
                health_ranges['poor'] += 1
        
        analytics['component_health_distribution'] = health_ranges
        
        return analytics

# Test and demonstration
async def test_adaptive_controller():
    """Test the adaptive controller system"""
    print("🧠 Testing CortexOS Adaptive Controller...")
    
    # Create configuration
    config = {
        'monitoring_interval': 5,  # 5 seconds for testing
        'adaptation_interval': 10,  # 10 seconds for testing
        'health_check_interval': 3,  # 3 seconds for testing
        'max_concurrent_adaptations': 2,
        'enable_auto_adaptation': True,
        'health_threshold': 0.7,
        'performance_threshold': 0.6,
        'stability_threshold': 0.8
    }
    
    # Initialize adaptive controller
    controller = AdaptiveController(config)
    
    # Mock component interfaces for testing
    class MockComponent:
        def __init__(self, name):
            self.name = name
            self.running = False
        
        async def start(self):
            self.running = True
            print(f"   Mock {self.name} started")
        
        async def stop(self):
            self.running = False
            print(f"   Mock {self.name} stopped")
    
    # Register mock components
    mock_components = [
        ComponentType.INFRASTRUCTURE,
        ComponentType.NEURAL_ENGINE,
        ComponentType.SYSTEM_MONITOR,
        ComponentType.MOOD_MODULATOR
    ]
    
    for comp_type in mock_components:
        mock_comp = MockComponent(comp_type.value)
        controller.component_manager.register_component(comp_type, mock_comp)
    
    try:
        # Start controller
        await controller.start()
        print("✅ Adaptive Controller started")
        
        # Wait for initialization
        await asyncio.sleep(3)
        
        # Display initial system status
        initial_status = controller.get_system_status()
        print(f"\n🏗️ Initial System Status:")
        print(f"   System State: {initial_status['system_state']}")
        print(f"   Control Mode: {initial_status['control_mode']}")
        print(f"   Overall Health: {initial_status['current_metrics']['overall_health']:.3f}")
        print(f"   Performance Score: {initial_status['current_metrics']['performance_score']:.3f}")
        print(f"   Running Components: {initial_status['components']['running']}/{initial_status['components']['total']}")
        
        # Display component details
        component_details = controller.get_component_details()
        print(f"\n🔧 Component Details:")
        for comp_name, details in component_details.items():
            print(f"   {comp_name}: {details['status']} (Health: {details['health_score']:.3f})")
        
        # Request manual adaptation
        print(f"\n🚀 Requesting manual adaptation...")
        
        adaptation_request = AdaptationRequest(
            request_id="test_adaptation_001",
            trigger_type="user",
            target_components=[ComponentType.NEURAL_ENGINE, ComponentType.SYSTEM_MONITOR],
            adaptation_type="optimization",
            parameters={'optimization_level': 'moderate'},
            context={'test': True}
        )
        
        await controller.request_adaptation(adaptation_request)
        
        # Wait for adaptation to process
        await asyncio.sleep(8)
        
        # Display status after adaptation
        adapted_status = controller.get_system_status()
        print(f"\n⚡ Status After Adaptation:")
        print(f"   System State: {adapted_status['system_state']}")
        print(f"   Overall Health: {adapted_status['current_metrics']['overall_health']:.3f}")
        print(f"   Performance Score: {adapted_status['current_metrics']['performance_score']:.3f}")
        print(f"   Active Adaptations: {adapted_status['adaptations']['active']}")
        
        # Test control mode changes
        print(f"\n🎛️ Testing control mode changes...")
        controller.set_control_mode(ControlMode.SEMI_AUTOMATIC)
        controller.set_adaptation_strategy(AdaptationStrategy.PROACTIVE)
        
        # Wait for auto-adaptation to potentially trigger
        print(f"\n⏳ Waiting for potential auto-adaptation...")
        await asyncio.sleep(15)
        
        # Display adaptation history
        history = controller.get_adaptation_history(1)
        print(f"\n📚 Adaptation History: {len(history)} adaptations")
        for adaptation in history:
            status = "✅" if adaptation['success'] else "❌"
            print(f"   {status} {adaptation['request_id']}: {len(adaptation['components_affected'])} components")
            if adaptation['performance_impact']:
                for comp, impact in adaptation['performance_impact'].items():
                    print(f"      {comp}: {impact:+.3f} impact")
        
        # Display system analytics
        print(f"\n📈 System Analytics:")
        analytics = controller.get_system_analytics()
        if analytics:
            print(f"   Average Health: {analytics['average_health']:.3f}")
            print(f"   Average Performance: {analytics['average_performance']:.3f}")
            print(f"   Average Stability: {analytics['average_stability']:.3f}")
            print(f"   Total Adaptations: {analytics['total_adaptations']}")
            print(f"   Success Rate: {analytics['adaptation_success_rate']:.1f}%")
            
            health_dist = analytics['component_health_distribution']
            print(f"   Component Health: Excellent({health_dist['excellent']}) Good({health_dist['good']}) Fair({health_dist['fair']}) Poor({health_dist['poor']})")
        
        # Display final system status
        final_status = controller.get_system_status()
        print(f"\n🎯 Final System Status:")
        print(f"   System State: {final_status['system_state']}")
        print(f"   Overall Health: {final_status['current_metrics']['overall_health']:.3f}")
        print(f"   Cognitive Load: {final_status['current_metrics']['cognitive_load']:.3f}")
        print(f"   Response Time: {final_status['current_metrics']['response_time']:.1f}ms")
        print(f"   Throughput: {final_status['current_metrics']['throughput']:.1f} ops/s")
        print(f"   Total Adaptations: {final_status['adaptations']['total_completed']}")
        
        print("\n✅ Adaptive Controller test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(test_adaptive_controller())

