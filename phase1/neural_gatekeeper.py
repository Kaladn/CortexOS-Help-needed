"""
phase1/neural_gatekeeper.py - CortexOS Neural Security Gatekeeper
Provides security and verification for neural operations, ensuring safety and validation.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Set, Optional
from collections import deque

# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
CONFIG_DIR = "{PATH_CONFIG_DIR}"

class NeuralGatekeeper:
    """
    Security and verification system for CortexOS neural architecture.
    
    Validates all neural operations against safety parameters, user intent,
    and operational context before allowing execution of sensitive operations.
    Implements the Agharmonic Law through strict interface contracts and
    resonance chain validation.
    """
    
    def __init__(self, security_threshold: float = 0.8, verification_timeout: float = 2.0):
        """
        Initialize the neural gatekeeper system.
        
        Args:
            security_threshold: Threshold for security validation (0.0-1.0)
            verification_timeout: Maximum time for verification in seconds
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🛡️ Initializing NeuralGatekeeper...")
        
        # Configuration
        self.security_threshold = security_threshold
        self.verification_timeout = verification_timeout
        self.max_verification_attempts = 3
        self.verification_cooldown = 1.0  # seconds
        
        # State tracking
        self.is_active = True
        self.verification_history = deque(maxlen=100)
        self.last_verification_time = 0
        self.verification_in_progress = False
        self.security_lock = threading.RLock()
        self.anomaly_counter = 0
        self.last_sync_time = time.time()
        self.sync_interval = 5.0  # seconds
        
        # Security policies
        self.security_policies = {}
        self.restricted_operations = set()
        self.authorized_sources = set()
        self.blocked_patterns = set()
        
        # Agharmonic compliance parameters
        self.input_frequency_range = (0.6, 1.4)  # Hz cognitive equivalent
        self.output_phase_alignment = 0.0
        self.resonance_threshold = 0.85  # Higher threshold for security
        self.degradation_levels = ["normal", "heightened", "restricted", "emergency"]
        self.current_degradation_level = "normal"
        
        # Component references
        self.trust_filter = None
        self.global_sync = None
        
        # Initialize security policies
        self._init_security_policies()
        
        self.logger.info("✅ NeuralGatekeeper initialized successfully")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def _init_security_policies(self):
        """Initialize default security policies"""
        self.security_policies = {
            'input_validation': {
                'max_input_size': 1024 * 1024,  # 1MB
                'allowed_types': ['str', 'dict', 'list', 'int', 'float', 'bool'],
                'forbidden_patterns': ['__', 'eval', 'exec', 'import', 'open', 'file'],
                'max_nesting_depth': 10
            },
            'operation_limits': {
                'max_operations_per_second': 100,
                'max_concurrent_operations': 10,
                'operation_timeout': 30.0
            },
            'resource_limits': {
                'max_memory_usage': 0.8,  # 80% of available
                'max_cpu_usage': 0.7,     # 70% of available
                'max_disk_usage': 0.9     # 90% of available
            }
        }
        
        # Initialize restricted operations
        self.restricted_operations.update([
            'system_shutdown',
            'format_drive',
            'delete_all_data',
            'modify_security_policies',
            'bypass_verification'
        ])
        
    def validate_input(self, input_package: Dict) -> bool:
        """
        Validate input package against security policies.
        
        Args:
            input_package: Input data package to validate
            
        Returns:
            True if input is valid and safe, False otherwise
        """
        try:
            with self.security_lock:
                # Check if gatekeeper is active
                if not self.is_active:
                    self.logger.warning("Gatekeeper inactive, rejecting input")
                    return False
                    
                # Check verification cooldown
                if time.time() - self.last_verification_time < self.verification_cooldown:
                    self.logger.debug("Verification cooldown active")
                    return False
                    
                # Start verification
                self.verification_in_progress = True
                verification_start = time.time()
                
                try:
                    # Basic structure validation
                    if not self._validate_structure(input_package):
                        return False
                        
                    # Content validation
                    if not self._validate_content(input_package):
                        return False
                        
                    # Security pattern validation
                    if not self._validate_security_patterns(input_package):
                        return False
                        
                    # Resource usage validation
                    if not self._validate_resource_usage(input_package):
                        return False
                        
                    # Agharmonic compliance validation
                    if not self._validate_agharmonic_compliance(input_package):
                        return False
                        
                    # Record successful verification
                    verification_time = time.time() - verification_start
                    self._record_verification(input_package, True, verification_time)
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Validation error: {e}")
                    self._record_verification(input_package, False, 0, str(e))
                    return False
                    
                finally:
                    self.verification_in_progress = False
                    self.last_verification_time = time.time()
                    
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
            
    def _validate_structure(self, input_package: Dict) -> bool:
        """Validate the basic structure of input package"""
        try:
            # Must be a dictionary
            if not isinstance(input_package, dict):
                self.logger.warning("Input package must be a dictionary")
                return False
                
            # Check required fields
            required_fields = ['data', 'type']
            for field in required_fields:
                if field not in input_package:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
                    
            # Check data size
            data_str = str(input_package.get('data', ''))
            max_size = self.security_policies['input_validation']['max_input_size']
            if len(data_str) > max_size:
                self.logger.warning(f"Input data too large: {len(data_str)} > {max_size}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Structure validation error: {e}")
            return False
            
    def _validate_content(self, input_package: Dict) -> bool:
        """Validate the content of input package"""
        try:
            data = input_package.get('data')
            data_type = input_package.get('type', 'unknown')
            
            # Check data type
            allowed_types = self.security_policies['input_validation']['allowed_types']
            if type(data).__name__ not in allowed_types:
                self.logger.warning(f"Invalid data type: {type(data).__name__}")
                return False
                
            # Check nesting depth for complex data
            if isinstance(data, (dict, list)):
                if self._get_nesting_depth(data) > self.security_policies['input_validation']['max_nesting_depth']:
                    self.logger.warning("Data nesting too deep")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Content validation error: {e}")
            return False
            
    def _validate_security_patterns(self, input_package: Dict) -> bool:
        """Validate against security patterns"""
        try:
            # Convert entire package to string for pattern matching
            package_str = str(input_package).lower()
            
            # Check forbidden patterns
            forbidden_patterns = self.security_policies['input_validation']['forbidden_patterns']
            for pattern in forbidden_patterns:
                if pattern in package_str:
                    self.logger.warning(f"Forbidden pattern detected: {pattern}")
                    return False
                    
            # Check blocked patterns
            for pattern in self.blocked_patterns:
                if pattern in package_str:
                    self.logger.warning(f"Blocked pattern detected: {pattern}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Security pattern validation error: {e}")
            return False
            
    def _validate_resource_usage(self, input_package: Dict) -> bool:
        """Validate resource usage constraints"""
        try:
            # Simple resource validation
            # In a real implementation, this would check actual system resources
            
            # Check operation rate limits
            recent_operations = sum(1 for v in self.verification_history 
                                  if time.time() - v['timestamp'] < 1.0)
            max_ops = self.security_policies['operation_limits']['max_operations_per_second']
            
            if recent_operations >= max_ops:
                self.logger.warning(f"Operation rate limit exceeded: {recent_operations}/{max_ops}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resource validation error: {e}")
            return False
            
    def _validate_agharmonic_compliance(self, input_package: Dict) -> bool:
        """Validate Agharmonic Law compliance"""
        try:
            # Check if operation is restricted
            operation = input_package.get('operation', 'unknown')
            if operation in self.restricted_operations:
                self.logger.warning(f"Restricted operation attempted: {operation}")
                return False
                
            # Check degradation level restrictions
            if self.current_degradation_level == "emergency":
                # Only allow critical operations in emergency mode
                critical_operations = ['status_check', 'emergency_shutdown', 'error_report']
                if operation not in critical_operations:
                    self.logger.warning(f"Non-critical operation blocked in emergency mode: {operation}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Agharmonic compliance validation error: {e}")
            return False
            
    def _get_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure"""
        try:
            if isinstance(data, dict):
                if not data:
                    return current_depth
                return max(self._get_nesting_depth(v, current_depth + 1) for v in data.values())
            elif isinstance(data, list):
                if not data:
                    return current_depth
                return max(self._get_nesting_depth(item, current_depth + 1) for item in data)
            else:
                return current_depth
                
        except Exception:
            return current_depth
            
    def _record_verification(self, input_package: Dict, success: bool, 
                           verification_time: float, error: str = None):
        """Record verification attempt"""
        try:
            verification_record = {
                'timestamp': time.time(),
                'success': success,
                'verification_time': verification_time,
                'input_type': input_package.get('type', 'unknown'),
                'input_size': len(str(input_package.get('data', ''))),
                'error': error,
                'degradation_level': self.current_degradation_level
            }
            
            self.verification_history.append(verification_record)
            
            # Update anomaly counter
            if not success:
                self.anomaly_counter += 1
            else:
                self.anomaly_counter = max(0, self.anomaly_counter - 1)
                
            # Check if degradation level should change
            self._update_degradation_level()
            
        except Exception as e:
            self.logger.error(f"Failed to record verification: {e}")
            
    def _update_degradation_level(self):
        """Update security degradation level based on recent activity"""
        try:
            # Calculate recent failure rate
            recent_verifications = [v for v in self.verification_history 
                                  if time.time() - v['timestamp'] < 60.0]  # Last minute
            
            if not recent_verifications:
                return
                
            failure_rate = sum(1 for v in recent_verifications if not v['success']) / len(recent_verifications)
            
            # Update degradation level
            if failure_rate > 0.5:
                self.current_degradation_level = "emergency"
            elif failure_rate > 0.3:
                self.current_degradation_level = "restricted"
            elif failure_rate > 0.1:
                self.current_degradation_level = "heightened"
            else:
                self.current_degradation_level = "normal"
                
        except Exception as e:
            self.logger.error(f"Failed to update degradation level: {e}")
            
    # Agharmonic Law Compliance Methods
    def harmonic_signature(self) -> Dict:
        """Establish frequency compatibility for Agharmonic Law compliance"""
        return {
            "base_frequency": 432.0,
            "security_frequency": 528.0,  # Higher frequency for security
            "variance_tolerance": 0.02,   # Stricter tolerance for security
            "phase_alignment": self.output_phase_alignment,
            "resonance_threshold": self.resonance_threshold
        }
        
    def interface_contract(self) -> Dict:
        """Define interface contract for external modules"""
        return {
            "allowed_calls": [
                "validate_input",
                "check_authorization",
                "get_security_status",
                "update_security_policies"
            ],
            "accepted_data_formats": [
                "input_package_v1",
                "security_query_v1",
                "authorization_request_v1"
            ],
            "security_level": "high",
            "compliance_level": "agharmonic_strict"
        }
        
    def cognitive_energy_flow(self, input_signal: float) -> float:
        """Normalize and validate cognitive energy flow"""
        try:
            # Apply security filtering to energy flow
            if not (self.input_frequency_range[0] <= input_signal <= self.input_frequency_range[1]):
                # Clamp to safe range
                input_signal = max(self.input_frequency_range[0], 
                                 min(self.input_frequency_range[1], input_signal))
                
            # Apply security dampening based on degradation level
            dampening_factors = {
                "normal": 1.0,
                "heightened": 0.8,
                "restricted": 0.5,
                "emergency": 0.2
            }
            
            dampening = dampening_factors.get(self.current_degradation_level, 0.2)
            return input_signal * dampening
            
        except Exception as e:
            self.logger.error(f"Cognitive energy flow error: {e}")
            return 0.0
            
    def sync_clock(self) -> float:
        """Synchronize with global clock"""
        try:
            if self.global_sync:
                return self.global_sync.get_sync_time()
            else:
                return time.time()
                
        except Exception as e:
            self.logger.error(f"Sync clock error: {e}")
            return time.time()
            
    def self_regulate(self) -> Dict:
        """Perform self-regulation and security monitoring"""
        try:
            # Calculate security metrics
            recent_failures = sum(1 for v in self.verification_history 
                                if not v['success'] and time.time() - v['timestamp'] < 300)
            
            avg_verification_time = 0.0
            if self.verification_history:
                times = [v['verification_time'] for v in self.verification_history if v['verification_time'] > 0]
                avg_verification_time = sum(times) / len(times) if times else 0.0
                
            regulation_status = {
                'security_level': self.current_degradation_level,
                'recent_failures': recent_failures,
                'anomaly_counter': self.anomaly_counter,
                'avg_verification_time': avg_verification_time,
                'is_active': self.is_active,
                'verification_in_progress': self.verification_in_progress
            }
            
            # Generate security recommendations
            recommendations = []
            if recent_failures > 10:
                recommendations.append('increase_security_threshold')
            if avg_verification_time > 1.0:
                recommendations.append('optimize_verification_process')
            if self.anomaly_counter > 20:
                recommendations.append('investigate_security_threats')
                
            return {
                'status': regulation_status,
                'recommendations': recommendations,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Self-regulation error: {e}")
            return {'error': str(e)}
            
    def graceful_fallback(self) -> Dict:
        """Implement graceful security fallback"""
        try:
            fallback_mode = {
                "status": "security_fallback",
                "actions": [
                    "increase_verification_threshold",
                    "reduce_operation_limits",
                    "enable_strict_filtering",
                    "activate_emergency_protocols"
                ],
                "restricted_operations": list(self.restricted_operations),
                "degradation_level": "emergency",
                "recovery_conditions": [
                    "anomaly_counter_below_threshold",
                    "verification_success_rate_improved",
                    "manual_security_override"
                ]
            }
            
            # Activate emergency mode
            self.current_degradation_level = "emergency"
            
            return fallback_mode
            
        except Exception as e:
            self.logger.error(f"Graceful fallback error: {e}")
            return {'error': str(e)}
            
    def resonance_chain_validator(self, resonance_data: Dict = None) -> bool:
        """Validate resonance chain for security compliance"""
        try:
            if not resonance_data:
                return True
                
            # Security-focused resonance validation
            required_security_fields = ['source', 'destination', 'security_level']
            for field in required_security_fields:
                if field not in resonance_data:
                    return False
                    
            # Validate security level
            security_level = resonance_data.get('security_level', 0)
            if security_level < self.resonance_threshold:
                return False
                
            # Check source authorization
            source = resonance_data.get('source')
            if source and source not in self.authorized_sources and self.authorized_sources:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resonance validation error: {e}")
            return False
            
    # Public Interface Methods
    def check_authorization(self, source: str, operation: str) -> bool:
        """Check if source is authorized for operation"""
        try:
            # Check if source is in authorized list
            if self.authorized_sources and source not in self.authorized_sources:
                return False
                
            # Check if operation is restricted
            if operation in self.restricted_operations:
                return False
                
            # Check degradation level restrictions
            if self.current_degradation_level == "emergency":
                critical_operations = ['status_check', 'emergency_shutdown', 'error_report']
                return operation in critical_operations
                
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization check error: {e}")
            return False
            
    def get_security_status(self) -> Dict:
        """Get current security status"""
        try:
            return {
                'is_active': self.is_active,
                'degradation_level': self.current_degradation_level,
                'security_threshold': self.security_threshold,
                'anomaly_counter': self.anomaly_counter,
                'verification_in_progress': self.verification_in_progress,
                'recent_verifications': len(self.verification_history),
                'authorized_sources': len(self.authorized_sources),
                'restricted_operations': len(self.restricted_operations)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {'error': str(e)}
            
    def add_authorized_source(self, source: str) -> bool:
        """Add an authorized source"""
        try:
            self.authorized_sources.add(source)
            self.logger.info(f"Added authorized source: {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add authorized source: {e}")
            return False
            
    def remove_authorized_source(self, source: str) -> bool:
        """Remove an authorized source"""
        try:
            self.authorized_sources.discard(source)
            self.logger.info(f"Removed authorized source: {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove authorized source: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS Neural Gatekeeper
    print("🛡️ Testing CortexOS Neural Gatekeeper...")
    
    gatekeeper = CortexOSNeuralGatekeeper()
    
    # Test gatekeeper startup
    if gatekeeper.start():
        print("✅ Neural Gatekeeper started successfully")
    
    # Test input validation
    test_inputs = [
        {"data": "Hello CortexOS", "source": "user", "input_type": "text"},
        {"data": {"command": "process", "params": [1, 2, 3]}, "source": "api", "input_type": "json"},
        {"data": "<script>alert('xss')</script>", "source": "web", "input_type": "html"},
        {"data": "SELECT * FROM users", "source": "database", "input_type": "sql"},
        {"data": {"valid": True, "content": "Clean data"}, "source": "trusted", "input_type": "json"}
    ]
    
    for test_input in test_inputs:
        is_valid = gatekeeper.validate_input(**test_input)
        status = "✅ PASSED" if is_valid else "❌ BLOCKED"
        print(f"{status} Input from {test_input['source']}: {test_input['input_type']}")
    
    # Test authorization
    gatekeeper.add_authorized_source("trusted_api")
    gatekeeper.add_authorized_source("admin_panel")
    
    auth_tests = [
        ("trusted_api", True),
        ("admin_panel", True),
        ("unknown_source", False),
        ("malicious_bot", False)
    ]
    
    for source, expected in auth_tests:
        is_authorized = gatekeeper.is_authorized(source)
        status = "✅" if is_authorized == expected else "❌"
        print(f"{status} Authorization test for {source}: {is_authorized}")
    
    # Test threat detection
    threat_tests = [
        "normal text input",
        "<script>malicious code</script>",
        "'; DROP TABLE users; --",
        "../../../../etc/passwd",
        "eval(malicious_code)"
    ]
    
    for threat_test in threat_tests:
        threat_level = gatekeeper.detect_threats(threat_test)
        status = "🔴 HIGH" if threat_level > 0.7 else "🟡 MEDIUM" if threat_level > 0.3 else "🟢 LOW"
        print(f"{status} Threat level for input: {threat_level:.2f}")
    
    # Test security metrics
    metrics = gatekeeper.get_security_metrics()
    print(f"✅ Security metrics - Processed: {metrics['total_processed']}, Blocked: {metrics['total_blocked']}")
    
    # Test gatekeeper status
    status = gatekeeper.get_status()
    print(f"✅ Gatekeeper status: {status['state']}, Active rules: {status['active_rules']}")
    
    # Shutdown
    gatekeeper.stop()
    print("✅ Neural Gatekeeper stopped")
    
    print("🛡️ Neural Gatekeeper test complete!")

