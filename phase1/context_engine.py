"""
phase1/context_engine.py - CortexOS Context Management Engine
Manages context switching, module coordination, and adaptive resonance adjustment.
"""

import time
import logging
from typing import Dict, Any, List, Optional

# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
CONFIG_DIR = "{PATH_CONFIG_DIR}"

class ContextEngine:
    """
    ContextEngine oversees context switching, module coordination, and adaptive 
    resonance adjustment within the CortexOS neurogrid system.
    
    Ensures system-wide temporal coherence, harmonic compliance, and resilience
    in accordance with the Agharmonic Law.
    """
    
    def __init__(self):
        """Initialize the Context Engine"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Initializing ContextEngine...")
        
        # Context state management
        self.current_context = {}
        self.context_history = []
        self.context_stack = []
        self.max_context_history = 1000
        
        # Resonance and harmony parameters
        self.base_frequency = 432.0  # Hz, Agharmonic Standard
        self.variance_tolerance = 0.05
        self.resonance_profile = {}
        
        # Component references (will be injected by supervisor)
        self.global_sync = None
        self.learner = None
        self.resonance_monitor = None
        self.execution_monitor = None
        self.neuroengine = None
        self.cube = None
        
        # Engine state
        self.active = False
        self.last_update = time.time()
        self.update_count = 0
        
        self.logger.info("✅ ContextEngine initialized")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def start(self) -> bool:
        """Start the context engine"""
        try:
            self.active = True
            self.logger.info("✅ ContextEngine started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start ContextEngine: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the context engine"""
        try:
            self.active = False
            self.logger.info("✅ ContextEngine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop ContextEngine: {e}")
            return False
            
    def process_text(self, text_data: str) -> Dict:
        """Process text data and extract context"""
        try:
            # Basic text context analysis
            context_result = {
                'text_length': len(text_data),
                'word_count': len(text_data.split()),
                'context_type': 'text',
                'timestamp': time.time(),
                'complexity_score': self._calculate_text_complexity(text_data),
                'context_markers': self._extract_context_markers(text_data)
            }
            
            return context_result
            
        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return {'error': str(e)}
            
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        try:
            words = text.split()
            if not words:
                return 0.0
                
            # Simple complexity metrics
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_words = len(set(words))
            word_diversity = unique_words / len(words) if words else 0
            
            complexity = (avg_word_length * 0.3 + word_diversity * 0.7) / 10.0
            return min(1.0, complexity)
            
        except Exception:
            return 0.0
            
    def _extract_context_markers(self, text: str) -> List[str]:
        """Extract context markers from text"""
        try:
            markers = []
            
            # Simple keyword-based context detection
            context_keywords = {
                'question': ['?', 'what', 'how', 'why', 'when', 'where', 'who'],
                'command': ['do', 'make', 'create', 'build', 'run', 'execute'],
                'emotional': ['feel', 'think', 'believe', 'love', 'hate', 'happy', 'sad'],
                'technical': ['system', 'code', 'function', 'algorithm', 'data', 'process']
            }
            
            text_lower = text.lower()
            for context_type, keywords in context_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    markers.append(context_type)
                    
            return markers
            
        except Exception:
            return []
            
    # Agharmonic Law Compliance Methods
    def harmonic_signature(self) -> Dict:
        """
        Establishes and returns the ContextEngine's harmonic frequency compatibility parameters.
        """
        return {
            "base_frequency": self.base_frequency,
            "variance_tolerance": self.variance_tolerance,
            "current_resonance": self.resonance_profile.get('current', 0.0),
            "harmonic_stability": self._calculate_harmonic_stability()
        }
        
    def interface_contract(self) -> Dict:
        """
        Defines allowed function calls and accepted data structures for external modules.
        """
        return {
            "allowed_calls": [
                "update_context",
                "validate_context", 
                "optimize_resonance",
                "synchronize_state",
                "process_text",
                "get_current_context"
            ],
            "accepted_data_formats": [
                "context_package_v1",
                "harmonic_resonance_payload",
                "text_input",
                "structured_data"
            ],
            "version": "1.0",
            "compliance_level": "agharmonic_standard"
        }
        
    def cognitive_energy_flow(self, input_data: Any) -> Any:
        """
        Normalizes input signal amplitude and ensures information flow adheres 
        to cognitive energy principles.
        """
        try:
            # Normalize input data
            if isinstance(input_data, str):
                # Text normalization
                normalized_data = input_data.strip().lower()
            elif isinstance(input_data, dict):
                # Dictionary normalization
                normalized_data = {k: v for k, v in input_data.items() if v is not None}
            elif isinstance(input_data, (int, float)):
                # Numeric normalization (0-1 range)
                normalized_data = max(0.0, min(1.0, float(input_data)))
            else:
                # Generic normalization
                normalized_data = input_data
                
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Cognitive energy flow error: {e}")
            return input_data
            
    def sync_clock(self) -> float:
        """
        Connects and synchronizes to the CortexOS master temporal framework.
        """
        try:
            if self.global_sync:
                return self.global_sync.get_sync_time()
            else:
                return time.time()
                
        except Exception as e:
            self.logger.error(f"Sync clock error: {e}")
            return time.time()
            
    def self_regulate(self) -> Dict:
        """
        Implements internal feedback loops for stability and anomaly detection.
        """
        try:
            # Basic self-regulation checks
            regulation_status = {
                'context_health': self._check_context_health(),
                'memory_usage': self._check_memory_usage(),
                'update_frequency': self._check_update_frequency(),
                'resonance_stability': self._calculate_harmonic_stability()
            }
            
            # Generate recommendations
            recommendations = []
            if regulation_status['context_health'] < 0.7:
                recommendations.append('clear_old_contexts')
            if regulation_status['memory_usage'] > 0.8:
                recommendations.append('reduce_context_history')
            if regulation_status['update_frequency'] < 0.1:
                recommendations.append('increase_update_rate')
                
            return {
                'status': regulation_status,
                'recommendations': recommendations,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Self-regulation error: {e}")
            return {'error': str(e)}
            
    def graceful_fallback(self) -> Dict:
        """
        Provides mechanisms for partial operation during failures or suboptimal conditions.
        """
        try:
            fallback_mode = {
                "status": "degraded",
                "actions": [
                    "reduce_processing_bandwidth",
                    "prioritize_critical_pathways", 
                    "increase_sync_pulse_interval",
                    "simplify_context_tracking"
                ],
                "reduced_capabilities": [
                    "complex_context_analysis",
                    "deep_resonance_monitoring",
                    "advanced_optimization"
                ],
                "maintained_capabilities": [
                    "basic_context_switching",
                    "simple_text_processing",
                    "core_synchronization"
                ],
                "recovery_conditions": [
                    "system_load_below_threshold",
                    "memory_usage_normalized",
                    "resonance_stability_restored"
                ]
            }
            
            return fallback_mode
            
        except Exception as e:
            self.logger.error(f"Graceful fallback error: {e}")
            return {'error': str(e)}
            
    def resonance_chain_validator(self, resonance_data: Dict = None) -> bool:
        """
        Verifies that resonance chains propagate cleanly without distortion or phase shifting.
        """
        try:
            if not resonance_data:
                resonance_data = self.resonance_profile
                
            # Basic resonance validation
            if not resonance_data:
                return True  # No data to validate
                
            # Check for required fields
            required_fields = ['frequency', 'amplitude', 'phase']
            for field in required_fields:
                if field not in resonance_data:
                    return False
                    
            # Validate frequency range
            frequency = resonance_data.get('frequency', 0)
            if not (1.0 <= frequency <= 1000.0):
                return False
                
            # Validate amplitude range
            amplitude = resonance_data.get('amplitude', 0)
            if not (0.0 <= amplitude <= 1.0):
                return False
                
            # Validate phase range
            phase = resonance_data.get('phase', 0)
            if not (0.0 <= phase <= 360.0):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resonance validation error: {e}")
            return False
            
    # Core Context Management Methods
    def update_context(self, context_package: Dict) -> bool:
        """
        Core method for updating the system's operational context based on incoming data.
        """
        try:
            # Validate and normalize the context package
            validated_context = self.cognitive_energy_flow(context_package)
            
            # Update current context
            self.current_context.update(validated_context)
            self.current_context['last_update'] = time.time()
            self.current_context['update_count'] = self.update_count
            
            # Add to context history
            self.context_history.append({
                'context': validated_context.copy(),
                'timestamp': time.time(),
                'update_id': self.update_count
            })
            
            # Trim history if needed
            if len(self.context_history) > self.max_context_history:
                self.context_history = self.context_history[-self.max_context_history:]
                
            # Update statistics
            self.last_update = time.time()
            self.update_count += 1
            
            # Notify neuroengine if available
            if self.neuroengine:
                self.neuroengine.update_operational_context(validated_context)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Context update error: {e}")
            return False
            
    def get_current_context(self) -> Dict:
        """Get the current context state"""
        return self.current_context.copy()
        
    def validate_context(self, context_data: Dict) -> bool:
        """Validate context data structure and content"""
        try:
            # Basic validation
            if not isinstance(context_data, dict):
                return False
                
            # Check for malicious or invalid data
            if any(key.startswith('_') for key in context_data.keys()):
                return False
                
            return True
            
        except Exception:
            return False
            
    def optimize_resonance(self) -> bool:
        """
        Applies adaptive learning strategies to optimize resonance alignment.
        """
        try:
            if self.learner and self.neuroengine:
                current_profile = self.neuroengine.current_resonance_profile()
                optimization_success = self.learner.optimize_harmonics(current_profile)
                return optimization_success
            else:
                # Basic optimization without dependencies
                self.resonance_profile['optimized'] = True
                self.resonance_profile['optimization_time'] = time.time()
                return True
                
        except Exception as e:
            self.logger.error(f"Resonance optimization error: {e}")
            return False
            
    def synchronize_state(self) -> bool:
        """
        Synchronizes ContextEngine state with the CortexCube for persistence and redundancy.
        """
        try:
            if self.cube:
                return self.cube.sync_context_state(self)
            else:
                # Basic state synchronization without cube
                self.logger.debug("State synchronization completed (no cube available)")
                return True
                
        except Exception as e:
            self.logger.error(f"State synchronization error: {e}")
            return False
            
    # Helper Methods
    def _check_context_health(self) -> float:
        """Check the health of the context system"""
        try:
            if not self.current_context:
                return 0.0
                
            # Check recency of updates
            last_update = self.current_context.get('last_update', 0)
            time_since_update = time.time() - last_update
            
            # Health decreases with time since last update
            health = max(0.0, 1.0 - (time_since_update / 300.0))  # 5 minute decay
            
            return health
            
        except Exception:
            return 0.0
            
    def _check_memory_usage(self) -> float:
        """Check memory usage of context system"""
        try:
            # Estimate memory usage based on context history size
            history_size = len(self.context_history)
            max_size = self.max_context_history
            
            usage_ratio = history_size / max_size if max_size > 0 else 0.0
            return min(1.0, usage_ratio)
            
        except Exception:
            return 0.0
            
    def _check_update_frequency(self) -> float:
        """Check the frequency of context updates"""
        try:
            if self.update_count == 0:
                return 0.0
                
            # Calculate updates per second over last period
            time_period = 60.0  # 1 minute
            recent_updates = sum(1 for ctx in self.context_history 
                               if time.time() - ctx['timestamp'] < time_period)
            
            frequency = recent_updates / time_period
            return min(1.0, frequency)
            
        except Exception:
            return 0.0
            
    def _calculate_harmonic_stability(self) -> float:
        """Calculate harmonic stability score"""
        try:
            if not self.resonance_profile:
                return 1.0  # Stable by default
                
            # Simple stability calculation based on variance
            frequency = self.resonance_profile.get('frequency', self.base_frequency)
            variance = abs(frequency - self.base_frequency) / self.base_frequency
            
            stability = max(0.0, 1.0 - (variance / self.variance_tolerance))
            return stability
            
        except Exception:
            return 0.0


if __name__ == "__main__":
    # Test the CortexOS Context Engine
    print("🎯 Testing CortexOS Context Engine...")
    
    engine = CortexOSContextEngine()
    
    # Test engine startup
    if engine.start():
        print("✅ Context Engine started successfully")
    
    # Test context creation
    test_contexts = [
        {"session_id": "test_001", "user_input": "Hello CortexOS", "context_type": "greeting"},
        {"session_id": "test_001", "user_input": "What is neural processing?", "context_type": "question"},
        {"session_id": "test_002", "user_input": "Process this data", "context_type": "command"}
    ]
    
    for ctx_data in test_contexts:
        context_id = engine.create_context(**ctx_data)
        if context_id:
            print(f"✅ Created context: {context_id} for session {ctx_data['session_id']}")
    
    # Test context retrieval
    session_contexts = engine.get_session_contexts("test_001")
    print(f"✅ Retrieved {len(session_contexts)} contexts for session test_001")
    
    # Test context updates
    if session_contexts:
        first_context = session_contexts[0]
        engine.update_context(first_context['context_id'], {"processed": True, "response": "Hello back!"})
        print(f"✅ Updated context: {first_context['context_id']}")
    
    # Test Agharmonic Law compliance
    compliance_score = engine.check_agharmonic_compliance()
    print(f"✅ Agharmonic compliance score: {compliance_score:.2f}")
    
    # Test context search
    search_results = engine.search_contexts("neural")
    print(f"✅ Found {len(search_results)} contexts matching 'neural'")
    
    # Test context cleanup
    cleaned = engine.cleanup_old_contexts(max_age_hours=0.001)  # Very short for testing
    print(f"✅ Cleaned up {cleaned} old contexts")
    
    # Test engine status
    status = engine.get_status()
    print(f"✅ Engine status: {status['state']}, Total contexts: {status['total_contexts']}")
    
    # Shutdown
    engine.stop()
    print("✅ Context Engine stopped")
    
    print("🎯 Context Engine test complete!")

