"""
phase2/resonance_monitor.py - CortexOS Resonance Stability Monitor
Monitors resonance stability and phase coherence across the neural architecture.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime

# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
RESONANCE_DATA_DIR = "{PATH_RESONANCE_DATA_DIR}"

class ResonanceMonitor:
    """
    Monitors resonance stability and phase coherence across the neural architecture.
    
    Detects and alerts on phase drift or resonance instability, implementing
    the Agharmonic Law through harmonic resonance monitoring and self-regulation.
    """
    
    def __init__(self, stability_threshold: float = 0.75):
        """
        Initialize the Resonance Monitor.
        
        Args:
            stability_threshold: Threshold for resonance stability (0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 Initializing ResonanceMonitor...")
        
        # Configuration
        self.stability_threshold = stability_threshold
        self.max_history_length = 1000
        self.check_interval = 1.0  # seconds
        
        # Monitoring state
        self.last_check_time = time.time()
        self.resonance_history = deque(maxlen=self.max_history_length)
        self.active_resonances = {}
        self.phase_drift_log = deque(maxlen=100)
        
        # Agharmonic Law compliance parameters
        self.input_frequency_range = (0.5, 2.0)  # GHz cognitive equivalent
        self.output_phase_alignment = 0.0
        self.resonance_threshold = 0.8
        self.harmonic_frequencies = [432.0, 528.0, 639.0, 741.0, 852.0]  # Hz
        
        # Stability tracking
        self.stability_score = 1.0
        self.phase_coherence = 1.0
        self.drift_rate = 0.0
        self.anomaly_count = 0
        
        # Performance metrics
        self.check_count = 0
        self.alert_count = 0
        self.correction_count = 0
        
        # Component references
        self.resonance_field = None
        self.global_sync = None
        
        self.logger.info("✅ ResonanceMonitor initialized")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def start_monitoring(self) -> bool:
        """Start resonance monitoring"""
        try:
            self.logger.info("🔍 Starting resonance monitoring...")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
            return False
            
    def stop_monitoring(self) -> bool:
        """Stop resonance monitoring"""
        try:
            self.logger.info("⏹️ Stopping resonance monitoring...")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop monitoring: {e}")
            return False
            
    def evaluate_system_resonance(self) -> Dict[str, Any]:
        """
        Evaluate overall system resonance stability.
        
        Returns:
            Comprehensive resonance evaluation report
        """
        try:
            current_time = time.time()
            self.check_count += 1
            
            # Collect resonance data from active sources
            resonance_data = self._collect_resonance_data()
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(resonance_data)
            
            # Detect phase drift
            phase_drift = self._detect_phase_drift(resonance_data)
            
            # Check for anomalies
            anomalies = self._detect_anomalies(resonance_data)
            
            # Generate stability report
            stability_report = {
                'timestamp': current_time,
                'stability_score': stability_metrics['overall_stability'],
                'phase_coherence': stability_metrics['phase_coherence'],
                'frequency_alignment': stability_metrics['frequency_alignment'],
                'phase_drift': phase_drift,
                'anomalies': anomalies,
                'active_resonances': len(self.active_resonances),
                'check_count': self.check_count,
                'recommendations': self._generate_recommendations(stability_metrics, anomalies)
            }
            
            # Update internal state
            self.stability_score = stability_metrics['overall_stability']
            self.phase_coherence = stability_metrics['phase_coherence']
            self.drift_rate = phase_drift.get('drift_rate', 0.0)
            
            # Add to history
            self.resonance_history.append(stability_report)
            
            # Check for alerts
            if stability_metrics['overall_stability'] < self.stability_threshold:
                self._trigger_stability_alert(stability_report)
                
            self.last_check_time = current_time
            
            return stability_report
            
        except Exception as e:
            self.logger.error(f"System resonance evaluation failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}
            
    def validate_resonance_chain(self, resonance_data: Dict = None) -> bool:
        """
        Verify that resonance chains propagate cleanly without distortion.
        
        Args:
            resonance_data: Optional resonance data to validate
            
        Returns:
            True if resonance chain is valid
        """
        try:
            if resonance_data is None:
                # Use current system resonance data
                resonance_data = self._collect_resonance_data()
                
            # Validate chain structure
            if not self._validate_chain_structure(resonance_data):
                return False
                
            # Check frequency coherence
            if not self._validate_frequency_coherence(resonance_data):
                return False
                
            # Verify phase alignment
            if not self._validate_phase_alignment(resonance_data):
                return False
                
            # Check for distortion
            if self._detect_signal_distortion(resonance_data):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resonance chain validation failed: {e}")
            return False
            
    def generate_execution_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive execution report for the monitoring system.
        
        Returns:
            Detailed execution report
        """
        try:
            current_time = time.time()
            
            # Calculate performance metrics
            uptime = current_time - (self.last_check_time - (self.check_count * self.check_interval))
            avg_check_interval = uptime / max(1, self.check_count)
            
            # Analyze recent stability trends
            recent_stability = self._analyze_stability_trends()
            
            # Calculate error rates
            error_rate = self.anomaly_count / max(1, self.check_count)
            
            execution_report = {
                'timestamp': current_time,
                'uptime': uptime,
                'total_checks': self.check_count,
                'avg_check_interval': avg_check_interval,
                'current_stability': self.stability_score,
                'current_phase_coherence': self.phase_coherence,
                'current_drift_rate': self.drift_rate,
                'anomaly_count': self.anomaly_count,
                'alert_count': self.alert_count,
                'correction_count': self.correction_count,
                'error_rate': error_rate,
                'stability_trends': recent_stability,
                'active_resonances': len(self.active_resonances),
                'history_length': len(self.resonance_history),
                'system_health': self._assess_system_health()
            }
            
            return execution_report
            
        except Exception as e:
            self.logger.error(f"Execution report generation failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}
            
    def _collect_resonance_data(self) -> Dict[str, Any]:
        """Collect current resonance data from all sources"""
        try:
            resonance_data = {
                'timestamp': time.time(),
                'sources': {},
                'field_data': {},
                'system_metrics': {}
            }
            
            # Collect from resonance field if available
            if self.resonance_field:
                try:
                    field_stats = self.resonance_field.get_field_stats()
                    resonance_data['field_data'] = field_stats
                    
                    # Get active neuron data
                    neuron_ids = self.resonance_field.registered_neuron_ids()
                    for neuron_id in neuron_ids[:10]:  # Limit to first 10 for performance
                        neuron_resonance = self._get_neuron_resonance(neuron_id)
                        if neuron_resonance:
                            resonance_data['sources'][neuron_id] = neuron_resonance
                            
                except Exception as e:
                    self.logger.warning(f"Failed to collect field data: {e}")
                    
            # Add system-level metrics
            resonance_data['system_metrics'] = {
                'stability_score': self.stability_score,
                'phase_coherence': self.phase_coherence,
                'drift_rate': self.drift_rate,
                'active_count': len(self.active_resonances)
            }
            
            return resonance_data
            
        except Exception as e:
            self.logger.error(f"Resonance data collection failed: {e}")
            return {'timestamp': time.time(), 'error': str(e)}
            
    def _get_neuron_resonance(self, neuron_id: str) -> Optional[Dict]:
        """Get resonance data for a specific neuron"""
        try:
            if not self.resonance_field:
                return None
                
            # Query neuron's current state
            neuron_data = self.resonance_field.resonance_buffer.get(neuron_id)
            if not neuron_data:
                return None
                
            return {
                'frequency': neuron_data.get('frequency', 0.0),
                'phase': neuron_data.get('phase', 0.0),
                'energy_level': neuron_data.get('energy_level', 0.0),
                'last_broadcast': neuron_data.get('last_broadcast', 0),
                'cluster_id': neuron_data.get('cluster_id'),
                'coords': neuron_data.get('coords', (0, 0, 0))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get neuron resonance for {neuron_id}: {e}")
            return None
            
    def _calculate_stability_metrics(self, resonance_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive stability metrics"""
        try:
            metrics = {
                'overall_stability': 1.0,
                'phase_coherence': 1.0,
                'frequency_alignment': 1.0,
                'energy_balance': 1.0,
                'temporal_consistency': 1.0
            }
            
            sources = resonance_data.get('sources', {})
            if not sources:
                return metrics
                
            # Calculate frequency alignment
            frequencies = [s.get('frequency', 0) for s in sources.values()]
            if frequencies:
                freq_variance = np.var(frequencies)
                freq_mean = np.mean(frequencies)
                if freq_mean > 0:
                    metrics['frequency_alignment'] = max(0.0, 1.0 - (freq_variance / freq_mean))
                    
            # Calculate phase coherence
            phases = [s.get('phase', 0) for s in sources.values()]
            if phases:
                phase_variance = np.var(phases)
                metrics['phase_coherence'] = max(0.0, 1.0 - (phase_variance / (2 * np.pi)))
                
            # Calculate energy balance
            energies = [s.get('energy_level', 0) for s in sources.values()]
            if energies:
                energy_variance = np.var(energies)
                energy_mean = np.mean(energies)
                if energy_mean > 0:
                    metrics['energy_balance'] = max(0.0, 1.0 - (energy_variance / energy_mean))
                    
            # Calculate temporal consistency
            current_time = time.time()
            broadcast_times = [s.get('last_broadcast', 0) for s in sources.values()]
            if broadcast_times:
                time_diffs = [current_time - t for t in broadcast_times if t > 0]
                if time_diffs:
                    avg_time_diff = np.mean(time_diffs)
                    metrics['temporal_consistency'] = max(0.0, 1.0 - min(1.0, avg_time_diff / 60.0))
                    
            # Calculate overall stability
            stability_components = [
                metrics['frequency_alignment'],
                metrics['phase_coherence'],
                metrics['energy_balance'],
                metrics['temporal_consistency']
            ]
            metrics['overall_stability'] = np.mean(stability_components)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Stability metrics calculation failed: {e}")
            return {'overall_stability': 0.0, 'phase_coherence': 0.0, 
                   'frequency_alignment': 0.0, 'energy_balance': 0.0, 
                   'temporal_consistency': 0.0}
                   
    def _detect_phase_drift(self, resonance_data: Dict) -> Dict[str, Any]:
        """Detect phase drift in the system"""
        try:
            drift_info = {
                'drift_detected': False,
                'drift_rate': 0.0,
                'affected_sources': [],
                'severity': 'none'
            }
            
            sources = resonance_data.get('sources', {})
            if len(sources) < 2:
                return drift_info
                
            # Compare current phases with historical data
            if len(self.resonance_history) > 1:
                previous_data = self.resonance_history[-1]
                prev_sources = previous_data.get('sources', {})
                
                phase_changes = []
                for source_id, current_data in sources.items():
                    if source_id in prev_sources:
                        current_phase = current_data.get('phase', 0)
                        prev_phase = prev_sources[source_id].get('phase', 0)
                        phase_change = abs(current_phase - prev_phase)
                        
                        # Normalize phase change to [0, π]
                        if phase_change > np.pi:
                            phase_change = 2 * np.pi - phase_change
                            
                        phase_changes.append(phase_change)
                        
                        # Check for significant drift
                        if phase_change > 0.5:  # Significant drift threshold
                            drift_info['affected_sources'].append(source_id)
                            
                if phase_changes:
                    avg_drift = np.mean(phase_changes)
                    time_diff = resonance_data['timestamp'] - previous_data['timestamp']
                    
                    if time_diff > 0:
                        drift_info['drift_rate'] = avg_drift / time_diff
                        
                    # Determine severity
                    if avg_drift > 1.0:
                        drift_info['severity'] = 'critical'
                        drift_info['drift_detected'] = True
                    elif avg_drift > 0.5:
                        drift_info['severity'] = 'high'
                        drift_info['drift_detected'] = True
                    elif avg_drift > 0.2:
                        drift_info['severity'] = 'moderate'
                        drift_info['drift_detected'] = True
                        
            # Log drift if detected
            if drift_info['drift_detected']:
                self.phase_drift_log.append({
                    'timestamp': resonance_data['timestamp'],
                    'drift_rate': drift_info['drift_rate'],
                    'severity': drift_info['severity'],
                    'affected_count': len(drift_info['affected_sources'])
                })
                
            return drift_info
            
        except Exception as e:
            self.logger.error(f"Phase drift detection failed: {e}")
            return {'drift_detected': False, 'error': str(e)}
            
    def _detect_anomalies(self, resonance_data: Dict) -> List[Dict]:
        """Detect anomalies in resonance patterns"""
        try:
            anomalies = []
            sources = resonance_data.get('sources', {})
            
            for source_id, source_data in sources.items():
                # Check frequency anomalies
                frequency = source_data.get('frequency', 0)
                if frequency < 100 or frequency > 2000:  # Outside normal range
                    anomalies.append({
                        'type': 'frequency_anomaly',
                        'source': source_id,
                        'value': frequency,
                        'severity': 'high' if frequency < 50 or frequency > 5000 else 'moderate'
                    })
                    
                # Check energy anomalies
                energy = source_data.get('energy_level', 0)
                if energy > 10.0:  # Abnormally high energy
                    anomalies.append({
                        'type': 'energy_spike',
                        'source': source_id,
                        'value': energy,
                        'severity': 'high' if energy > 50.0 else 'moderate'
                    })
                elif energy < 0:  # Negative energy (impossible)
                    anomalies.append({
                        'type': 'negative_energy',
                        'source': source_id,
                        'value': energy,
                        'severity': 'critical'
                    })
                    
                # Check temporal anomalies
                last_broadcast = source_data.get('last_broadcast', 0)
                if last_broadcast > 0:
                    time_since_broadcast = time.time() - last_broadcast
                    if time_since_broadcast > 300:  # No broadcast for 5 minutes
                        anomalies.append({
                            'type': 'stale_broadcast',
                            'source': source_id,
                            'value': time_since_broadcast,
                            'severity': 'moderate'
                        })
                        
            # Update anomaly counter
            self.anomaly_count += len(anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return [{'type': 'detection_error', 'error': str(e)}]
            
    def _generate_recommendations(self, stability_metrics: Dict, anomalies: List) -> List[str]:
        """Generate recommendations based on stability analysis"""
        recommendations = []
        
        try:
            # Stability-based recommendations
            if stability_metrics['overall_stability'] < 0.5:
                recommendations.append('critical_stability_intervention_required')
            elif stability_metrics['overall_stability'] < 0.7:
                recommendations.append('increase_monitoring_frequency')
                
            if stability_metrics['frequency_alignment'] < 0.6:
                recommendations.append('recalibrate_frequency_sources')
                
            if stability_metrics['phase_coherence'] < 0.6:
                recommendations.append('synchronize_phase_alignment')
                
            if stability_metrics['energy_balance'] < 0.6:
                recommendations.append('rebalance_energy_distribution')
                
            # Anomaly-based recommendations
            critical_anomalies = [a for a in anomalies if a.get('severity') == 'critical']
            if critical_anomalies:
                recommendations.append('immediate_anomaly_investigation')
                
            high_anomalies = [a for a in anomalies if a.get('severity') == 'high']
            if len(high_anomalies) > 3:
                recommendations.append('system_health_check')
                
            # Performance recommendations
            if self.check_count > 100 and self.anomaly_count / self.check_count > 0.1:
                recommendations.append('review_system_configuration')
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ['error_in_recommendation_system']
            
    def _trigger_stability_alert(self, stability_report: Dict):
        """Trigger stability alert"""
        try:
            self.alert_count += 1
            
            alert_message = (
                f"STABILITY ALERT: System stability below threshold "
                f"({stability_report['stability_score']:.3f} < {self.stability_threshold})"
            )
            
            self.logger.warning(alert_message)
            
            # Additional alert actions could be implemented here
            # (e.g., notifications, automatic corrections, etc.)
            
        except Exception as e:
            self.logger.error(f"Failed to trigger stability alert: {e}")
            
    def _validate_chain_structure(self, resonance_data: Dict) -> bool:
        """Validate resonance chain structure"""
        try:
            sources = resonance_data.get('sources', {})
            
            # Must have at least one source
            if not sources:
                return False
                
            # Each source must have required fields
            required_fields = ['frequency', 'phase', 'energy_level']
            for source_data in sources.values():
                if not all(field in source_data for field in required_fields):
                    return False
                    
            return True
            
        except Exception:
            return False
            
    def _validate_frequency_coherence(self, resonance_data: Dict) -> bool:
        """Validate frequency coherence across sources"""
        try:
            sources = resonance_data.get('sources', {})
            frequencies = [s.get('frequency', 0) for s in sources.values()]
            
            if not frequencies:
                return True
                
            # Check if frequencies are within acceptable range
            for freq in frequencies:
                if not (50 <= freq <= 2000):  # Reasonable frequency range
                    return False
                    
            # Check frequency variance
            if len(frequencies) > 1:
                freq_variance = np.var(frequencies)
                freq_mean = np.mean(frequencies)
                if freq_mean > 0 and freq_variance / freq_mean > 0.5:  # High variance
                    return False
                    
            return True
            
        except Exception:
            return False
            
    def _validate_phase_alignment(self, resonance_data: Dict) -> bool:
        """Validate phase alignment across sources"""
        try:
            sources = resonance_data.get('sources', {})
            phases = [s.get('phase', 0) for s in sources.values()]
            
            if len(phases) < 2:
                return True
                
            # Check phase variance
            phase_variance = np.var(phases)
            if phase_variance > np.pi:  # High phase variance
                return False
                
            return True
            
        except Exception:
            return False
            
    def _detect_signal_distortion(self, resonance_data: Dict) -> bool:
        """Detect signal distortion in resonance data"""
        try:
            sources = resonance_data.get('sources', {})
            
            for source_data in sources.values():
                # Check for impossible values
                frequency = source_data.get('frequency', 0)
                energy = source_data.get('energy_level', 0)
                phase = source_data.get('phase', 0)
                
                # Frequency distortion
                if frequency <= 0 or frequency > 10000:
                    return True
                    
                # Energy distortion
                if energy < 0 or energy > 100:
                    return True
                    
                # Phase distortion
                if phase < 0 or phase > 2 * np.pi:
                    return True
                    
            return False
            
        except Exception:
            return True  # Assume distortion if we can't check
            
    def _analyze_stability_trends(self) -> Dict[str, Any]:
        """Analyze recent stability trends"""
        try:
            if len(self.resonance_history) < 5:
                return {'trend': 'insufficient_data'}
                
            # Get recent stability scores
            recent_scores = [r.get('stability_score', 0) for r in list(self.resonance_history)[-10:]]
            
            # Calculate trend
            if len(recent_scores) >= 2:
                trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                
                if trend_slope > 0.01:
                    trend = 'improving'
                elif trend_slope < -0.01:
                    trend = 'degrading'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
                
            return {
                'trend': trend,
                'trend_slope': trend_slope if 'trend_slope' in locals() else 0.0,
                'recent_avg': np.mean(recent_scores),
                'recent_variance': np.var(recent_scores),
                'sample_size': len(recent_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Stability trend analysis failed: {e}")
            return {'trend': 'error', 'error': str(e)}
            
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        try:
            health_score = 0
            max_score = 5
            
            # Stability score contribution
            if self.stability_score > 0.8:
                health_score += 2
            elif self.stability_score > 0.6:
                health_score += 1
                
            # Phase coherence contribution
            if self.phase_coherence > 0.8:
                health_score += 1
                
            # Error rate contribution
            if self.check_count > 0:
                error_rate = self.anomaly_count / self.check_count
                if error_rate < 0.05:
                    health_score += 1
                    
            # Alert rate contribution
            if self.check_count > 0:
                alert_rate = self.alert_count / self.check_count
                if alert_rate < 0.1:
                    health_score += 1
                    
            # Determine health status
            health_ratio = health_score / max_score
            
            if health_ratio >= 0.8:
                return 'excellent'
            elif health_ratio >= 0.6:
                return 'good'
            elif health_ratio >= 0.4:
                return 'fair'
            elif health_ratio >= 0.2:
                return 'poor'
            else:
                return 'critical'
                
        except Exception:
            return 'unknown'
            
    # Agharmonic Law Compliance Methods
    def harmonic_signature(self) -> Dict[str, Any]:
        """Establish frequency compatibility for this module"""
        return {
            "input_range": self.input_frequency_range,
            "output_phase": self.output_phase_alignment,
            "threshold": self.resonance_threshold,
            "harmonic_frequencies": self.harmonic_frequencies,
            "stability_threshold": self.stability_threshold
        }
        
    def interface_contract(self) -> Dict[str, List[str]]:
        """Define interface contract for external modules"""
        return {
            "inputs": ["resonance_id", "phase", "strength", "check_interval"],
            "outputs": ["stability_report", "phase_drift", "active_resonances"],
            "methods": [
                "evaluate_system_resonance",
                "validate_resonance_chain", 
                "generate_execution_report"
            ]
        }
        
    def cognitive_energy_flow(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signals based on amplitude and entropy"""
        try:
            if signal.size == 0:
                return np.array([])
                
            # Normalize signal to maintain balanced information flow
            signal_min = signal.min()
            signal_max = signal.max()
            
            if signal_max - signal_min == 0:
                return np.full_like(signal, 0.5)
                
            normalized = (signal - signal_min) / (signal_max - signal_min)
            
            # Scale to range [0.05, 0.95] to avoid extremes
            return normalized * 0.9 + 0.05
            
        except Exception as e:
            self.logger.error(f"Cognitive energy flow normalization failed: {e}")
            return signal
            
    def sync_clock(self, global_clock: Any = None) -> bool:
        """Connect to master temporal framework"""
        try:
            if global_clock and hasattr(global_clock, 'get_time'):
                synchronized_time = global_clock.get_time()
                self.last_check_time = synchronized_time
                return True
            else:
                self.last_check_time = time.time()
                return True
                
        except Exception as e:
            self.logger.error(f"Clock synchronization failed: {e}")
            return False
            
    def self_regulate(self) -> Dict[str, Any]:
        """Perform self-regulation and internal monitoring"""
        try:
            regulation_status = {
                'monitoring_health': self._assess_system_health(),
                'check_frequency': self.check_count / max(1, time.time() - self.last_check_time + self.check_count),
                'error_rate': self.anomaly_count / max(1, self.check_count),
                'alert_rate': self.alert_count / max(1, self.check_count),
                'memory_usage': len(self.resonance_history) / self.max_history_length,
                'drift_log_size': len(self.phase_drift_log)
            }
            
            # Generate self-regulation recommendations
            recommendations = []
            if regulation_status['error_rate'] > 0.2:
                recommendations.append('increase_validation_strictness')
            if regulation_status['memory_usage'] > 0.9:
                recommendations.append('clear_old_history')
            if regulation_status['check_frequency'] < 0.1:
                recommendations.append('increase_check_frequency')
                
            return {
                'status': regulation_status,
                'recommendations': recommendations,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Self-regulation failed: {e}")
            return {'error': str(e)}
            
    def graceful_fallback(self) -> Dict[str, Any]:
        """Implement graceful fallback for monitoring failures"""
        try:
            fallback_mode = {
                "status": "monitoring_fallback",
                "actions": [
                    "reduce_check_frequency",
                    "simplify_stability_calculations",
                    "disable_complex_analysis",
                    "maintain_basic_monitoring"
                ],
                "reduced_capabilities": [
                    "detailed_anomaly_detection",
                    "complex_trend_analysis",
                    "advanced_recommendations"
                ],
                "maintained_capabilities": [
                    "basic_stability_monitoring",
                    "simple_phase_tracking",
                    "alert_generation"
                ]
            }
            
            # Reduce monitoring complexity
            self.check_interval = min(5.0, self.check_interval * 2)
            
            return fallback_mode
            
        except Exception as e:
            self.logger.error(f"Graceful fallback failed: {e}")
            return {'error': str(e)}
            
    def resonance_chain_validator(self, resonance_data: Dict = None) -> bool:
        """Validate resonance chain integrity"""
        return self.validate_resonance_chain(resonance_data)
        
    # Public Interface Methods
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        return {
            'check_count': self.check_count,
            'alert_count': self.alert_count,
            'correction_count': self.correction_count,
            'anomaly_count': self.anomaly_count,
            'current_stability': self.stability_score,
            'current_phase_coherence': self.phase_coherence,
            'current_drift_rate': self.drift_rate,
            'active_resonances': len(self.active_resonances),
            'history_length': len(self.resonance_history),
            'drift_log_length': len(self.phase_drift_log),
            'system_health': self._assess_system_health()
        }


if __name__ == "__main__":
    # Test the CortexOS Resonance Monitor
    print("📊 Testing CortexOS Resonance Monitor...")
    
    monitor = CortexOSResonanceMonitor()
    
    # Test monitor startup
    if monitor.start():
        print("✅ Resonance Monitor started successfully")
    
    # Test resonance monitoring with various scenarios
    test_resonances = [
        # Normal resonance
        {"frequency": 440.0, "amplitude": 0.8, "phase": 0.0, "stability": 0.9, "source": "neuron_1"},
        # High amplitude
        {"frequency": 880.0, "amplitude": 1.2, "phase": 0.1, "stability": 0.8, "source": "neuron_2"},
        # Phase drift
        {"frequency": 440.0, "amplitude": 0.8, "phase": 0.5, "stability": 0.7, "source": "neuron_1"},
        # Low stability (anomaly)
        {"frequency": 660.0, "amplitude": 0.9, "phase": 0.2, "stability": 0.3, "source": "neuron_3"},
        # Normal again
        {"frequency": 440.0, "amplitude": 0.8, "phase": 0.0, "stability": 0.9, "source": "neuron_1"}
    ]
    
    for i, resonance in enumerate(test_resonances):
        result = monitor.monitor_resonance(resonance)
        status = "✅ STABLE" if result else "⚠️ UNSTABLE"
        print(f"{status} Resonance {i+1}: {resonance['frequency']} Hz, Stability: {resonance['stability']}")
        time.sleep(0.1)  # Small delay between measurements
    
    # Test anomaly detection
    anomalies = monitor.detect_anomalies()
    print(f"✅ Detected {len(anomalies)} anomalies")
    
    # Test stability analysis
    stability_report = monitor.analyze_stability()
    print(f"✅ Stability analysis - Score: {stability_report['overall_stability']:.2f}, Trend: {stability_report['trend']}")
    
    # Test phase coherence
    coherence = monitor.check_phase_coherence()
    print(f"✅ Phase coherence: {coherence:.2f}")
    
    # Test drift detection
    drift_rate = monitor.detect_phase_drift()
    print(f"✅ Phase drift rate: {drift_rate:.4f} rad/s")
    
    # Test alert generation
    alerts = monitor.generate_alerts()
    if alerts:
        print(f"⚠️ Generated {len(alerts)} alerts:")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['type']}: {alert['message']}")
    else:
        print("✅ No alerts generated - system stable")
    
    # Test monitoring statistics
    stats = monitor.get_monitoring_stats()
    print(f"✅ Monitor stats - Checks: {stats['check_count']}, Alerts: {stats['alert_count']}, Health: {stats['system_health']:.2f}")
    
    # Test monitor status
    status = monitor.get_status()
    print(f"✅ Monitor status: {status['state']}, Active resonances: {status['active_resonances']}")
    
    # Shutdown
    monitor.stop()
    print("✅ Resonance Monitor stopped")
    
    print("📊 Resonance Monitor test complete!")

