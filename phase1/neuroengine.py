"""
phase1/neuroengine.py - CortexOS Core Neural Processing Engine
Main neural processing engine that orchestrates cognitive operations.
"""

import time
import queue
import threading
import logging
from typing import Dict, Any, Optional, List

# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
TEMP_DIR = "{PATH_TEMP_DIR}"

class NeuroEngine:
    """
    Core neural processing engine for CortexOS.
    Orchestrates all cognitive operations and manages neural state transitions.
    """
    
    def __init__(self, resonance_field=None, context_engine=None, gatekeeper=None):
        """Initialize the NeuroEngine"""
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Initializing NeuroEngine...")
        
        # Core dependencies
        self.resonance_field = resonance_field
        self.context_engine = context_engine
        self.gatekeeper = gatekeeper
        
        # Processing queues
        self.input_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Engine state
        self.running = False
        self.processing_thread = None
        self.input_thread = None
        self.output_thread = None
        self.last_cycle_time = 0
        self.cycle_count = 0
        
        # Engine statistics
        self.engine_state = {
            "status": "initialized",
            "mood": "neutral",
            "active_resonances": 0,
            "cycle_frequency": 0,
            "total_processed": 0,
            "errors": 0
        }
        
        # Processing configuration
        self.max_queue_size = 1000
        self.processing_timeout = 30.0
        self.cycle_delay = 0.01  # 10ms between cycles
        
        self.logger.info("✅ NeuroEngine initialized successfully")
        
    def start(self) -> bool:
        """Start the neural engine processing threads"""
        try:
            if self.running:
                self.logger.warning("NeuroEngine already running")
                return False
                
            self.running = True
            self.engine_state["status"] = "starting"
            
            # Start processing threads
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self.output_thread = threading.Thread(target=self._output_loop, daemon=True)
            
            self.processing_thread.start()
            self.input_thread.start()
            self.output_thread.start()
            
            self.engine_state["status"] = "running"
            self.logger.info("✅ NeuroEngine started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start NeuroEngine: {e}")
            self.engine_state["status"] = "error"
            return False
            
    def stop(self) -> bool:
        """Stop the neural engine processing threads"""
        try:
            if not self.running:
                self.logger.warning("NeuroEngine not running")
                return False
                
            self.running = False
            self.engine_state["status"] = "stopping"
            
            # Wait for threads to terminate
            threads = [self.processing_thread, self.input_thread, self.output_thread]
            for thread in threads:
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
                    
            self.engine_state["status"] = "stopped"
            self.logger.info("✅ NeuroEngine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop NeuroEngine: {e}")
            return False
            
    def process_input(self, input_data: Any, input_type: str = "text", 
                     metadata: Dict = None) -> Optional[str]:
        """
        Process input data through the neural engine.
        
        Args:
            input_data: Input data to process
            input_type: Type of input ('text', 'image', 'structured', etc.)
            metadata: Additional metadata for processing
            
        Returns:
            Request ID for tracking the processing
        """
        try:
            if not self.running:
                self.logger.warning("Cannot process input: NeuroEngine not running")
                return None
                
            # Generate request ID
            request_id = f"req-{int(time.time())}-{hash(str(input_data)) % 10000}"
            
            # Create input package
            input_package = {
                'request_id': request_id,
                'data': input_data,
                'type': input_type,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'status': 'queued'
            }
            
            # Add to input queue
            if self.input_queue.qsize() < self.max_queue_size:
                self.input_queue.put(input_package)
                self.logger.debug(f"Input queued: {request_id}")
                return request_id
            else:
                self.logger.warning("Input queue full, rejecting request")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to process input: {e}")
            return None
            
    def get_output(self, request_id: str = None, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get processed output from the engine.
        
        Args:
            request_id: Specific request ID to get output for
            timeout: Maximum time to wait for output
            
        Returns:
            Output package or None if not available
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    output_package = self.output_queue.get(timeout=0.1)
                    
                    # If specific request ID requested, check if it matches
                    if request_id and output_package.get('request_id') != request_id:
                        # Put it back and continue looking
                        self.output_queue.put(output_package)
                        continue
                        
                    return output_package
                    
                except queue.Empty:
                    continue
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get output: {e}")
            return None
            
    def _input_loop(self):
        """Input processing loop"""
        while self.running:
            try:
                # Get input from queue
                input_package = self.input_queue.get(timeout=1.0)
                
                # Validate input through gatekeeper
                if self.gatekeeper and not self.gatekeeper.validate_input(input_package):
                    self.logger.warning(f"Input rejected by gatekeeper: {input_package['request_id']}")
                    continue
                    
                # Move to processing queue
                input_package['status'] = 'processing'
                self.processing_queue.put(input_package)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Input loop error: {e}")
                self.engine_state["errors"] += 1
                
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                cycle_start = time.time()
                
                # Get item from processing queue
                try:
                    input_package = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    time.sleep(self.cycle_delay)
                    continue
                    
                # Process the input
                output_package = self._process_neural_data(input_package)
                
                # Add to output queue
                if output_package:
                    self.output_queue.put(output_package)
                    
                # Update cycle statistics
                cycle_time = time.time() - cycle_start
                self.last_cycle_time = cycle_time
                self.cycle_count += 1
                self.engine_state["total_processed"] += 1
                
                # Update cycle frequency
                if self.cycle_count % 100 == 0:
                    self.engine_state["cycle_frequency"] = 1.0 / max(cycle_time, 0.001)
                    
                # Brief delay to prevent CPU overload
                time.sleep(self.cycle_delay)
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                self.engine_state["errors"] += 1
                
    def _output_loop(self):
        """Output management loop"""
        while self.running:
            try:
                # Clean up old outputs to prevent memory buildup
                if self.output_queue.qsize() > self.max_queue_size:
                    try:
                        # Remove oldest output
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Output loop error: {e}")
                
    def _process_neural_data(self, input_package: Dict) -> Optional[Dict]:
        """
        Core neural data processing method.
        
        Args:
            input_package: Input data package to process
            
        Returns:
            Processed output package
        """
        try:
            request_id = input_package['request_id']
            input_data = input_package['data']
            input_type = input_package['type']
            
            # Initialize output package
            output_package = {
                'request_id': request_id,
                'input_type': input_type,
                'timestamp': time.time(),
                'processing_time': 0,
                'status': 'processing'
            }
            
            processing_start = time.time()
            
            # Process based on input type
            if input_type == "text":
                result = self._process_text_input(input_data)
            elif input_type == "structured":
                result = self._process_structured_input(input_data)
            elif input_type == "neural":
                result = self._process_neural_input(input_data)
            else:
                result = self._process_generic_input(input_data)
                
            # Finalize output package
            output_package.update({
                'result': result,
                'processing_time': time.time() - processing_start,
                'status': 'completed'
            })
            
            return output_package
            
        except Exception as e:
            self.logger.error(f"Neural processing error: {e}")
            return {
                'request_id': input_package.get('request_id', 'unknown'),
                'error': str(e),
                'status': 'error',
                'timestamp': time.time()
            }
            
    def _process_text_input(self, text_data: str) -> Dict:
        """Process text input through neural pathways"""
        try:
            # Basic text processing
            result = {
                'type': 'text_analysis',
                'length': len(text_data),
                'word_count': len(text_data.split()),
                'processed_text': text_data.lower().strip()
            }
            
            # Apply context engine if available
            if self.context_engine:
                context_result = self.context_engine.process_text(text_data)
                result['context'] = context_result
                
            # Apply resonance field if available
            if self.resonance_field:
                resonance_result = self.resonance_field.analyze_text(text_data)
                result['resonance'] = resonance_result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return {'error': str(e)}
            
    def _process_structured_input(self, structured_data: Dict) -> Dict:
        """Process structured data input"""
        try:
            result = {
                'type': 'structured_analysis',
                'keys': list(structured_data.keys()) if isinstance(structured_data, dict) else [],
                'data_type': type(structured_data).__name__,
                'processed_data': structured_data
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Structured processing error: {e}")
            return {'error': str(e)}
            
    def _process_neural_input(self, neural_data: Any) -> Dict:
        """Process neural-specific input"""
        try:
            result = {
                'type': 'neural_processing',
                'neural_signature': hash(str(neural_data)) % 10000,
                'processed_neural': neural_data
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Neural processing error: {e}")
            return {'error': str(e)}
            
    def _process_generic_input(self, input_data: Any) -> Dict:
        """Process generic input data"""
        try:
            result = {
                'type': 'generic_processing',
                'data_type': type(input_data).__name__,
                'data_size': len(str(input_data)),
                'processed_generic': str(input_data)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generic processing error: {e}")
            return {'error': str(e)}
            
    def get_engine_stats(self) -> Dict:
        """Get current engine statistics"""
        return {
            **self.engine_state,
            'input_queue_size': self.input_queue.qsize(),
            'processing_queue_size': self.processing_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'last_cycle_time': self.last_cycle_time,
            'cycle_count': self.cycle_count,
            'running': self.running
        }
        
    def clear_queues(self) -> bool:
        """Clear all processing queues"""
        try:
            # Clear all queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
                
            while not self.processing_queue.empty():
                self.processing_queue.get_nowait()
                
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
                
            self.logger.info("✅ All queues cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear queues: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS NeuroEngine
    print("🧠 Testing CortexOS NeuroEngine...")
    
    engine = CortexOSNeuroEngine()
    
    # Test engine startup
    if engine.start():
        print("✅ NeuroEngine started successfully")
    
    # Test data processing
    test_data = [
        {"type": "text", "content": "Hello CortexOS", "priority": 1},
        {"type": "pattern", "content": [1, 2, 3, 4, 5], "priority": 2},
        {"type": "signal", "content": {"frequency": 440, "amplitude": 0.8}, "priority": 1}
    ]
    
    for data in test_data:
        result = engine.process_data(data)
        if result:
            print(f"✅ Processed {data['type']} data successfully")
    
    # Test batch processing
    batch_data = [
        {"content": f"batch_item_{i}", "type": "batch"} for i in range(5)
    ]
    
    batch_results = engine.process_batch(batch_data)
    print(f"✅ Batch processing completed: {len(batch_results)} results")
    
    # Test engine status
    status = engine.get_status()
    print(f"✅ Engine status: {status['state']}, Queue size: {status['queue_size']}")
    
    # Test performance metrics
    metrics = engine.get_performance_metrics()
    print(f"✅ Performance - Processed: {metrics['total_processed']}, Rate: {metrics['processing_rate']:.2f}/sec")
    
    # Let it process for a moment
    time.sleep(2)
    
    # Test queue clearing
    if engine.clear_queues():
        print("✅ Queues cleared successfully")
    
    # Shutdown
    engine.stop()
    print("✅ NeuroEngine stopped")
    
    print("🧠 NeuroEngine test complete!")

