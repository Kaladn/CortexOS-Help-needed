#!/usr/bin/env python3
"""
CortexOS Phase 5: Performance Tracker
Advanced performance monitoring and optimization system
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
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class MetricTrend(Enum):
    """Metric trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"

class OptimizationAction(Enum):
    """Optimization action types"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"
    CACHE_OPTIMIZE = "cache_optimize"
    MEMORY_CLEANUP = "memory_cleanup"
    CPU_THROTTLE = "cpu_throttle"
    NO_ACTION = "no_action"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_id: str
    name: str
    value: float
    unit: str
    timestamp: datetime
    baseline: Optional[float] = None
    target: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_id: str
    trend_direction: MetricTrend
    trend_strength: float  # 0.0 to 1.0
    slope: float
    correlation: float
    prediction_next: Optional[float] = None
    confidence: float = 0.0

@dataclass
class PerformanceProfile:
    """Component performance profile"""
    component_id: str
    component_name: str
    performance_level: PerformanceLevel
    overall_score: float
    metrics: Dict[str, PerformanceMetric]
    trends: Dict[str, PerformanceTrend]
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    component_id: str
    action: OptimizationAction
    description: str
    expected_improvement: float
    confidence: float
    priority: int  # 1-10, 10 being highest
    estimated_effort: str  # "low", "medium", "high"
    prerequisites: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    timestamp: datetime
    overall_performance_level: PerformanceLevel
    overall_score: float
    component_profiles: Dict[str, PerformanceProfile]
    system_bottlenecks: List[str]
    optimization_recommendations: List[OptimizationRecommendation]
    performance_summary: Dict[str, Any] = field(default_factory=dict)

class TrendAnalyzer:
    """Performance trend analysis engine"""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
    
    def analyze_trend(self, values: List[float], timestamps: List[datetime]) -> PerformanceTrend:
        """Analyze performance trend from historical data"""
        if len(values) < self.min_data_points:
            return PerformanceTrend(
                metric_id="unknown",
                trend_direction=MetricTrend.STABLE,
                trend_strength=0.0,
                slope=0.0,
                correlation=0.0
            )
        
        try:
            # Convert timestamps to numeric values (seconds since first timestamp)
            time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # Calculate linear regression
            slope, correlation = self._calculate_linear_regression(time_numeric, values)
            
            # Determine trend direction and strength
            trend_direction, trend_strength = self._classify_trend(slope, correlation, values)
            
            # Predict next value
            prediction_next = None
            confidence = 0.0
            
            if len(values) >= 5:
                prediction_next, confidence = self._predict_next_value(time_numeric, values)
            
            return PerformanceTrend(
                metric_id="analyzed",
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                correlation=correlation,
                prediction_next=prediction_next,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return PerformanceTrend(
                metric_id="error",
                trend_direction=MetricTrend.STABLE,
                trend_strength=0.0,
                slope=0.0,
                correlation=0.0
            )
    
    def _calculate_linear_regression(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and correlation"""
        n = len(x_values)
        
        if n < 2:
            return 0.0, 0.0
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate slope and correlation
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)
        
        if x_variance == 0:
            return 0.0, 0.0
        
        slope = numerator / x_variance
        
        if y_variance == 0:
            correlation = 0.0
        else:
            correlation = numerator / (x_variance * y_variance) ** 0.5
        
        return slope, correlation
    
    def _classify_trend(self, slope: float, correlation: float, values: List[float]) -> Tuple[MetricTrend, float]:
        """Classify trend direction and strength"""
        # Calculate coefficient of variation for volatility
        if len(values) > 1:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            cv = std_val / mean_val if mean_val != 0 else 0
        else:
            cv = 0
        
        # High volatility indicates volatile trend
        if cv > 0.3:
            return MetricTrend.VOLATILE, cv
        
        # Use correlation strength to determine trend confidence
        trend_strength = abs(correlation)
        
        # Classify based on slope and correlation
        if abs(correlation) < 0.3:  # Weak correlation
            return MetricTrend.STABLE, trend_strength
        elif slope > 0:
            return MetricTrend.IMPROVING, trend_strength
        else:
            return MetricTrend.DEGRADING, trend_strength
    
    def _predict_next_value(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Predict next value using linear regression"""
        try:
            slope, correlation = self._calculate_linear_regression(x_values, y_values)
            
            # Predict next value
            next_x = x_values[-1] + (x_values[-1] - x_values[-2]) if len(x_values) >= 2 else x_values[-1] + 1
            x_mean = sum(x_values) / len(x_values)
            y_mean = sum(y_values) / len(y_values)
            
            prediction = y_mean + slope * (next_x - x_mean)
            
            # Confidence based on correlation strength
            confidence = abs(correlation)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error predicting next value: {e}")
            return 0.0, 0.0

class BottleneckDetector:
    """System bottleneck detection engine"""
    
    def __init__(self):
        self.detection_rules = {
            'cpu_bottleneck': self._detect_cpu_bottleneck,
            'memory_bottleneck': self._detect_memory_bottleneck,
            'io_bottleneck': self._detect_io_bottleneck,
            'network_bottleneck': self._detect_network_bottleneck,
            'concurrency_bottleneck': self._detect_concurrency_bottleneck
        }
    
    def detect_bottlenecks(self, metrics: Dict[str, PerformanceMetric]) -> List[str]:
        """Detect system bottlenecks from metrics"""
        bottlenecks = []
        
        for bottleneck_type, detection_func in self.detection_rules.items():
            try:
                if detection_func(metrics):
                    bottlenecks.append(bottleneck_type)
            except Exception as e:
                logger.error(f"Error detecting {bottleneck_type}: {e}")
        
        return bottlenecks
    
    def _detect_cpu_bottleneck(self, metrics: Dict[str, PerformanceMetric]) -> bool:
        """Detect CPU bottleneck"""
        cpu_metrics = [m for m in metrics.values() if 'cpu' in m.metric_id.lower()]
        
        for metric in cpu_metrics:
            if metric.value > 85.0:  # High CPU usage
                return True
        
        return False
    
    def _detect_memory_bottleneck(self, metrics: Dict[str, PerformanceMetric]) -> bool:
        """Detect memory bottleneck"""
        memory_metrics = [m for m in metrics.values() if 'memory' in m.metric_id.lower()]
        
        for metric in memory_metrics:
            if 'usage' in metric.metric_id and metric.value > 90.0:  # High memory usage
                return True
        
        return False
    
    def _detect_io_bottleneck(self, metrics: Dict[str, PerformanceMetric]) -> bool:
        """Detect I/O bottleneck"""
        io_metrics = [m for m in metrics.values() if any(term in m.metric_id.lower() for term in ['disk', 'io', 'read', 'write'])]
        
        # Simple heuristic: high disk usage or low I/O throughput
        for metric in io_metrics:
            if 'usage' in metric.metric_id and metric.value > 95.0:
                return True
        
        return False
    
    def _detect_network_bottleneck(self, metrics: Dict[str, PerformanceMetric]) -> bool:
        """Detect network bottleneck"""
        network_metrics = [m for m in metrics.values() if 'network' in m.metric_id.lower()]
        
        # This would require more sophisticated analysis in a real system
        return False
    
    def _detect_concurrency_bottleneck(self, metrics: Dict[str, PerformanceMetric]) -> bool:
        """Detect concurrency bottleneck"""
        # Look for high wait times, lock contention, etc.
        concurrency_metrics = [m for m in metrics.values() if any(term in m.metric_id.lower() for term in ['wait', 'lock', 'queue'])]
        
        for metric in concurrency_metrics:
            if metric.value > metric.threshold_warning if metric.threshold_warning else 100:
                return True
        
        return False

class OptimizationEngine:
    """Performance optimization recommendation engine"""
    
    def __init__(self):
        self.optimization_rules = {
            'cpu_bottleneck': self._optimize_cpu,
            'memory_bottleneck': self._optimize_memory,
            'io_bottleneck': self._optimize_io,
            'network_bottleneck': self._optimize_network,
            'concurrency_bottleneck': self._optimize_concurrency
        }
    
    def generate_recommendations(self, profile: PerformanceProfile, bottlenecks: List[str]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck in self.optimization_rules:
                try:
                    recs = self.optimization_rules[bottleneck](profile)
                    recommendations.extend(recs)
                except Exception as e:
                    logger.error(f"Error generating recommendations for {bottleneck}: {e}")
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _optimize_cpu(self, profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate CPU optimization recommendations"""
        recommendations = []
        
        # High CPU usage recommendations
        cpu_metrics = [m for m in profile.metrics.values() if 'cpu' in m.metric_id.lower()]
        
        for metric in cpu_metrics:
            if metric.value > 85.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_opt_{int(time.time())}",
                    component_id=profile.component_id,
                    action=OptimizationAction.SCALE_UP,
                    description="Scale up CPU resources or optimize CPU-intensive operations",
                    expected_improvement=0.3,
                    confidence=0.8,
                    priority=8,
                    estimated_effort="medium"
                ))
        
        return recommendations
    
    def _optimize_memory(self, profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        memory_metrics = [m for m in profile.metrics.values() if 'memory' in m.metric_id.lower()]
        
        for metric in memory_metrics:
            if metric.value > 90.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"mem_opt_{int(time.time())}",
                    component_id=profile.component_id,
                    action=OptimizationAction.MEMORY_CLEANUP,
                    description="Implement memory cleanup and garbage collection optimization",
                    expected_improvement=0.25,
                    confidence=0.7,
                    priority=9,
                    estimated_effort="low"
                ))
        
        return recommendations
    
    def _optimize_io(self, profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate I/O optimization recommendations"""
        recommendations = []
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"io_opt_{int(time.time())}",
            component_id=profile.component_id,
            action=OptimizationAction.CACHE_OPTIMIZE,
            description="Implement I/O caching and buffering strategies",
            expected_improvement=0.4,
            confidence=0.6,
            priority=7,
            estimated_effort="medium"
        ))
        
        return recommendations
    
    def _optimize_network(self, profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate network optimization recommendations"""
        recommendations = []
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"net_opt_{int(time.time())}",
            component_id=profile.component_id,
            action=OptimizationAction.REBALANCE,
            description="Optimize network traffic distribution and connection pooling",
            expected_improvement=0.2,
            confidence=0.5,
            priority=6,
            estimated_effort="high"
        ))
        
        return recommendations
    
    def _optimize_concurrency(self, profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate concurrency optimization recommendations"""
        recommendations = []
        
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"conc_opt_{int(time.time())}",
            component_id=profile.component_id,
            action=OptimizationAction.REBALANCE,
            description="Optimize thread pool sizes and reduce lock contention",
            expected_improvement=0.35,
            confidence=0.7,
            priority=8,
            estimated_effort="medium"
        ))
        
        return recommendations

class PerformanceTracker:
    """Advanced performance monitoring and optimization system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.trend_analyzer = TrendAnalyzer()
        self.bottleneck_detector = BottleneckDetector()
        self.optimization_engine = OptimizationEngine()
        
        # Performance data storage
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.performance_profiles = {}
        self.performance_reports = deque(maxlen=100)
        
        # Configuration
        self.analysis_interval = self.config.get('analysis_interval', 60)  # seconds
        self.trend_window = self.config.get('trend_window', 3600)  # 1 hour
        self.enable_predictions = self.config.get('enable_predictions', True)
        self.enable_optimization = self.config.get('enable_optimization', True)
        
        # State
        self.running = False
        self.analysis_task = None
        
        logger.info("Performance Tracker initialized")
    
    async def start(self):
        """Start performance tracking"""
        try:
            self.running = True
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            logger.info("Performance Tracker started")
            
        except Exception as e:
            logger.error(f"Error starting Performance Tracker: {e}")
            raise
    
    async def stop(self):
        """Stop performance tracking"""
        try:
            self.running = False
            
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Performance Tracker stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Performance Tracker: {e}")
    
    def record_metric(self, component_id: str, metric: PerformanceMetric):
        """Record performance metric"""
        try:
            metric_key = f"{component_id}_{metric.metric_id}"
            self.metric_history[metric_key].append(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    def record_metrics_batch(self, component_id: str, metrics: List[PerformanceMetric]):
        """Record batch of performance metrics"""
        for metric in metrics:
            self.record_metric(component_id, metric)
    
    async def _analysis_loop(self):
        """Main performance analysis loop"""
        logger.info("Performance analysis loop started")
        
        while self.running:
            try:
                # Analyze performance for all components
                await self._analyze_performance()
                
                # Generate performance report
                report = self._generate_performance_report()
                self.performance_reports.append(report)
                
                # Wait for next analysis
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Performance analysis loop stopped")
    
    async def _analyze_performance(self):
        """Analyze performance for all components"""
        try:
            # Group metrics by component
            component_metrics = defaultdict(dict)
            
            for metric_key, metric_history in self.metric_history.items():
                if not metric_history:
                    continue
                
                # Parse component_id from metric_key
                parts = metric_key.split('_', 1)
                if len(parts) != 2:
                    continue
                
                component_id, metric_id = parts
                latest_metric = metric_history[-1]
                component_metrics[component_id][metric_id] = latest_metric
            
            # Analyze each component
            for component_id, metrics in component_metrics.items():
                profile = await self._analyze_component_performance(component_id, metrics)
                self.performance_profiles[component_id] = profile
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    async def _analyze_component_performance(self, component_id: str, metrics: Dict[str, PerformanceMetric]) -> PerformanceProfile:
        """Analyze performance for a single component"""
        try:
            # Calculate performance score
            performance_score = self._calculate_performance_score(metrics)
            
            # Determine performance level
            performance_level = self._classify_performance_level(performance_score)
            
            # Analyze trends
            trends = {}
            for metric_id, metric in metrics.items():
                trend = await self._analyze_metric_trend(component_id, metric_id)
                if trend:
                    trends[metric_id] = trend
            
            # Detect bottlenecks
            bottlenecks = self.bottleneck_detector.detect_bottlenecks(metrics)
            
            # Generate recommendations
            recommendations = []
            if self.enable_optimization:
                profile_temp = PerformanceProfile(
                    component_id=component_id,
                    component_name=f"Component {component_id}",
                    performance_level=performance_level,
                    overall_score=performance_score,
                    metrics=metrics,
                    trends=trends,
                    bottlenecks=bottlenecks
                )
                
                optimization_recs = self.optimization_engine.generate_recommendations(profile_temp, bottlenecks)
                recommendations = [rec.description for rec in optimization_recs]
            
            return PerformanceProfile(
                component_id=component_id,
                component_name=f"Component {component_id}",
                performance_level=performance_level,
                overall_score=performance_score,
                metrics=metrics,
                trends=trends,
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing component {component_id} performance: {e}")
            return PerformanceProfile(
                component_id=component_id,
                component_name=f"Component {component_id}",
                performance_level=PerformanceLevel.FAIR,
                overall_score=0.5,
                metrics=metrics,
                trends={}
            )
    
    def _calculate_performance_score(self, metrics: Dict[str, PerformanceMetric]) -> float:
        """Calculate overall performance score"""
        if not metrics:
            return 0.5
        
        scores = []
        
        for metric in metrics.values():
            # Calculate metric score based on thresholds
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                score = 0.0
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                score = 0.3
            elif metric.target and metric.value <= metric.target:
                score = 1.0
            else:
                # Linear interpolation between warning and target
                if metric.threshold_warning and metric.target:
                    ratio = (metric.value - metric.target) / (metric.threshold_warning - metric.target)
                    score = max(0.3, 1.0 - ratio * 0.7)
                else:
                    score = 0.7  # Default decent score
            
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _classify_performance_level(self, score: float) -> PerformanceLevel:
        """Classify performance level from score"""
        if score >= 0.9:
            return PerformanceLevel.EXCELLENT
        elif score >= 0.7:
            return PerformanceLevel.GOOD
        elif score >= 0.5:
            return PerformanceLevel.FAIR
        elif score >= 0.3:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    async def _analyze_metric_trend(self, component_id: str, metric_id: str) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric"""
        try:
            metric_key = f"{component_id}_{metric_id}"
            metric_history = self.metric_history[metric_key]
            
            if len(metric_history) < 5:
                return None
            
            # Get recent history within trend window
            cutoff_time = datetime.now() - timedelta(seconds=self.trend_window)
            recent_metrics = [m for m in metric_history if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 5:
                return None
            
            values = [m.value for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            trend = self.trend_analyzer.analyze_trend(values, timestamps)
            trend.metric_id = metric_id
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {component_id}_{metric_id}: {e}")
            return None
    
    def _generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            timestamp = datetime.now()
            
            # Calculate overall performance
            if self.performance_profiles:
                overall_score = sum(p.overall_score for p in self.performance_profiles.values()) / len(self.performance_profiles)
                overall_level = self._classify_performance_level(overall_score)
            else:
                overall_score = 0.5
                overall_level = PerformanceLevel.FAIR
            
            # Collect system bottlenecks
            system_bottlenecks = []
            for profile in self.performance_profiles.values():
                system_bottlenecks.extend(profile.bottlenecks)
            
            # Remove duplicates
            system_bottlenecks = list(set(system_bottlenecks))
            
            # Generate optimization recommendations
            optimization_recommendations = []
            if self.enable_optimization:
                for profile in self.performance_profiles.values():
                    recs = self.optimization_engine.generate_recommendations(profile, profile.bottlenecks)
                    optimization_recommendations.extend(recs)
            
            # Performance summary
            performance_summary = {
                'total_components': len(self.performance_profiles),
                'excellent_components': sum(1 for p in self.performance_profiles.values() if p.performance_level == PerformanceLevel.EXCELLENT),
                'good_components': sum(1 for p in self.performance_profiles.values() if p.performance_level == PerformanceLevel.GOOD),
                'fair_components': sum(1 for p in self.performance_profiles.values() if p.performance_level == PerformanceLevel.FAIR),
                'poor_components': sum(1 for p in self.performance_profiles.values() if p.performance_level == PerformanceLevel.POOR),
                'critical_components': sum(1 for p in self.performance_profiles.values() if p.performance_level == PerformanceLevel.CRITICAL),
                'total_bottlenecks': len(system_bottlenecks),
                'total_recommendations': len(optimization_recommendations)
            }
            
            return PerformanceReport(
                report_id=f"perf_report_{int(time.time())}",
                timestamp=timestamp,
                overall_performance_level=overall_level,
                overall_score=overall_score,
                component_profiles=self.performance_profiles.copy(),
                system_bottlenecks=system_bottlenecks,
                optimization_recommendations=optimization_recommendations,
                performance_summary=performance_summary
            )
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return PerformanceReport(
                report_id=f"error_report_{int(time.time())}",
                timestamp=datetime.now(),
                overall_performance_level=PerformanceLevel.FAIR,
                overall_score=0.5,
                component_profiles={},
                system_bottlenecks=[],
                optimization_recommendations=[]
            )
    
    def get_current_performance(self) -> Optional[PerformanceReport]:
        """Get most recent performance report"""
        return self.performance_reports[-1] if self.performance_reports else None
    
    def get_component_performance(self, component_id: str) -> Optional[PerformanceProfile]:
        """Get performance profile for specific component"""
        return self.performance_profiles.get(component_id)
    
    def get_performance_history(self, duration: int = 3600) -> List[PerformanceReport]:
        """Get performance report history"""
        cutoff_time = datetime.now() - timedelta(seconds=duration)
        return [report for report in self.performance_reports if report.timestamp >= cutoff_time]
    
    def get_metric_trend(self, component_id: str, metric_id: str) -> Optional[PerformanceTrend]:
        """Get trend analysis for specific metric"""
        profile = self.performance_profiles.get(component_id)
        if profile and metric_id in profile.trends:
            return profile.trends[metric_id]
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get performance tracker status"""
        current_report = self.get_current_performance()
        
        return {
            'running': self.running,
            'analysis_interval': self.analysis_interval,
            'trend_window': self.trend_window,
            'enable_predictions': self.enable_predictions,
            'enable_optimization': self.enable_optimization,
            'tracked_components': len(self.performance_profiles),
            'total_metrics': sum(len(history) for history in self.metric_history.values()),
            'reports_generated': len(self.performance_reports),
            'current_performance_level': current_report.overall_performance_level.value if current_report else 'unknown',
            'current_performance_score': current_report.overall_score if current_report else 0.0
        }

# Test and demonstration
async def test_performance_tracker():
    """Test the performance tracker system"""
    print("🧠 Testing CortexOS Performance Tracker...")
    
    # Create configuration
    config = {
        'analysis_interval': 5,  # 5 seconds for testing
        'trend_window': 60,      # 1 minute for testing
        'enable_predictions': True,
        'enable_optimization': True
    }
    
    # Initialize tracker
    tracker = PerformanceTracker(config)
    
    try:
        # Start tracking
        await tracker.start()
        print("✅ Performance Tracker started")
        
        # Simulate performance metrics
        print("\n📊 Simulating performance metrics...")
        
        components = ['neural_engine', 'memory_system', 'io_processor']
        
        for i in range(10):
            timestamp = datetime.now()
            
            for component_id in components:
                # CPU metrics
                cpu_metric = PerformanceMetric(
                    metric_id='cpu_usage',
                    name='CPU Usage',
                    value=50 + i * 3 + np.random.normal(0, 5),  # Gradually increasing
                    unit='percent',
                    timestamp=timestamp,
                    threshold_warning=80.0,
                    threshold_critical=95.0
                )
                
                # Memory metrics
                memory_metric = PerformanceMetric(
                    metric_id='memory_usage',
                    name='Memory Usage',
                    value=60 + i * 2 + np.random.normal(0, 3),
                    unit='percent',
                    timestamp=timestamp,
                    threshold_warning=85.0,
                    threshold_critical=95.0
                )
                
                # Throughput metrics
                throughput_metric = PerformanceMetric(
                    metric_id='throughput',
                    name='Processing Throughput',
                    value=1000 - i * 10 + np.random.normal(0, 20),  # Gradually decreasing
                    unit='ops/sec',
                    timestamp=timestamp,
                    target=1000.0,
                    threshold_warning=500.0,
                    threshold_critical=200.0
                )
                
                tracker.record_metrics_batch(component_id, [cpu_metric, memory_metric, throughput_metric])
            
            await asyncio.sleep(1)
        
        # Wait for analysis
        print("⏳ Analyzing performance data...")
        await asyncio.sleep(8)
        
        # Get current performance report
        current_report = tracker.get_current_performance()
        if current_report:
            print(f"\n📈 Performance Report:")
            print(f"   Overall Level: {current_report.overall_performance_level.value}")
            print(f"   Overall Score: {current_report.overall_score:.3f}")
            print(f"   Components Tracked: {len(current_report.component_profiles)}")
            print(f"   System Bottlenecks: {len(current_report.system_bottlenecks)}")
            print(f"   Optimization Recommendations: {len(current_report.optimization_recommendations)}")
            
            # Display component performance
            print(f"\n🔧 Component Performance:")
            for comp_id, profile in current_report.component_profiles.items():
                print(f"   {comp_id}: {profile.performance_level.value} (Score: {profile.overall_score:.3f})")
                if profile.bottlenecks:
                    print(f"      Bottlenecks: {', '.join(profile.bottlenecks)}")
                if profile.recommendations:
                    print(f"      Recommendations: {len(profile.recommendations)}")
            
            # Display trends
            print(f"\n📊 Performance Trends:")
            for comp_id, profile in current_report.component_profiles.items():
                for metric_id, trend in profile.trends.items():
                    print(f"   {comp_id}.{metric_id}: {trend.trend_direction.value} (Strength: {trend.trend_strength:.3f})")
            
            # Display optimization recommendations
            if current_report.optimization_recommendations:
                print(f"\n💡 Top Optimization Recommendations:")
                for rec in current_report.optimization_recommendations[:3]:
                    print(f"   {rec.action.value}: {rec.description}")
                    print(f"      Expected Improvement: {rec.expected_improvement:.1%}, Priority: {rec.priority}/10")
        
        # Display tracker status
        print(f"\n🔧 Tracker Status:")
        status = tracker.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Performance Tracker test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await tracker.stop()

if __name__ == "__main__":
    asyncio.run(test_performance_tracker())

