"""
infrastructure/sync_manager.py - CortexOS Global Synchronization Manager
Manages temporal synchronization across all CortexOS components.
"""

import time
import threading
import logging
from typing import Dict, List, Callable, Any
from collections import defaultdict, deque

# Path placeholders
LOGS_DIR = "{PATH_LOGS_DIR}"
CONFIG_DIR = "{PATH_CONFIG_DIR}"

class GlobalSyncManager:
    """
    Global synchronization manager for CortexOS.
    Coordinates timing and synchronization across all neural components.
    """
    
    def __init__(self, base_frequency: float = 1.0):
        """Initialize the sync manager"""
        self.base_frequency = base_frequency  # Hz
        self.sync_clock = 0.0
        self.running = False
        self.sync_thread = None
        
        # Component registration
        self.registered_components = {}
        self.sync_callbacks = defaultdict(list)
        
        # Timing management
        self.cycle_history = deque(maxlen=1000)
        self.last_cycle_time = 0.0
        self.cycle_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def register_component(self, component_id: str, component: Any, 
                          sync_callback: Callable = None) -> bool:
        """Register a component for synchronization"""
        try:
            with self.lock:
                self.registered_components[component_id] = {
                    'component': component,
                    'last_sync': 0.0,
                    'sync_count': 0,
                    'active': True
                }
                
                if sync_callback:
                    self.sync_callbacks[component_id].append(sync_callback)
                    
            self.logger.info(f"✅ Component registered: {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register component {component_id}: {e}")
            return False
            
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from synchronization"""
        try:
            with self.lock:
                if component_id in self.registered_components:
                    del self.registered_components[component_id]
                    
                if component_id in self.sync_callbacks:
                    del self.sync_callbacks[component_id]
                    
            self.logger.info(f"✅ Component unregistered: {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister component {component_id}: {e}")
            return False
            
    def start(self) -> bool:
        """Start the global synchronization system"""
        try:
            if self.running:
                self.logger.warning("Sync manager already running")
                return False
                
            self.running = True
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            
            self.logger.info("✅ Global sync manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start sync manager: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the global synchronization system"""
        try:
            self.running = False
            
            if self.sync_thread and self.sync_thread.is_alive():
                self.sync_thread.join(timeout=5.0)
                
            self.logger.info("✅ Global sync manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop sync manager: {e}")
            return False
            
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                cycle_start = time.time()
                
                # Update sync clock
                self.sync_clock = cycle_start
                
                # Synchronize all registered components
                self._synchronize_components()
                
                # Update cycle statistics
                cycle_duration = time.time() - cycle_start
                self.cycle_history.append({
                    'cycle': self.cycle_count,
                    'start_time': cycle_start,
                    'duration': cycle_duration,
                    'components_synced': len(self.registered_components)
                })
                
                self.last_cycle_time = cycle_duration
                self.cycle_count += 1
                
                # Sleep to maintain frequency
                sleep_time = max(0, (1.0 / self.base_frequency) - cycle_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                time.sleep(0.1)  # Brief pause before retry
                
    def _synchronize_components(self):
        """Synchronize all registered components"""
        with self.lock:
            for component_id, component_info in self.registered_components.items():
                if not component_info['active']:
                    continue
                    
                try:
                    # Update component sync info
                    component_info['last_sync'] = self.sync_clock
                    component_info['sync_count'] += 1
                    
                    # Call sync callbacks
                    for callback in self.sync_callbacks.get(component_id, []):
                        callback(self.sync_clock, self.cycle_count)
                        
                except Exception as e:
                    self.logger.error(f"Failed to sync component {component_id}: {e}")
                    
    def get_sync_time(self) -> float:
        """Get the current synchronization time"""
        return self.sync_clock
        
    def get_cycle_count(self) -> int:
        """Get the current cycle count"""
        return self.cycle_count
        
    def set_component_active(self, component_id: str, active: bool) -> bool:
        """Set a component's active status"""
        try:
            with self.lock:
                if component_id in self.registered_components:
                    self.registered_components[component_id]['active'] = active
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set component active status: {e}")
            return False
            
    def get_sync_stats(self) -> Dict:
        """Get synchronization statistics"""
        try:
            with self.lock:
                active_components = sum(1 for info in self.registered_components.values() 
                                      if info['active'])
                
                recent_cycles = list(self.cycle_history)[-10:] if self.cycle_history else []
                avg_cycle_time = (sum(c['duration'] for c in recent_cycles) / len(recent_cycles) 
                                if recent_cycles else 0.0)
                
                return {
                    'running': self.running,
                    'sync_clock': self.sync_clock,
                    'cycle_count': self.cycle_count,
                    'base_frequency': self.base_frequency,
                    'registered_components': len(self.registered_components),
                    'active_components': active_components,
                    'last_cycle_time': self.last_cycle_time,
                    'average_cycle_time': avg_cycle_time,
                    'total_cycles': len(self.cycle_history)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get sync stats: {e}")
            return {}
            
    def wait_for_sync(self, timeout: float = 1.0) -> bool:
        """Wait for the next synchronization cycle"""
        try:
            start_cycle = self.cycle_count
            start_time = time.time()
            
            while (self.cycle_count <= start_cycle and 
                   time.time() - start_time < timeout):
                time.sleep(0.001)  # 1ms sleep
                
            return self.cycle_count > start_cycle
            
        except Exception as e:
            self.logger.error(f"Wait for sync failed: {e}")
            return False
            
    def force_sync(self) -> bool:
        """Force an immediate synchronization of all components"""
        try:
            if not self.running:
                return False
                
            self.sync_clock = time.time()
            self._synchronize_components()
            
            self.logger.info("✅ Forced synchronization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Force sync failed: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS Sync Manager
    print("🔄 Testing CortexOS Sync Manager...")
    
    sync_manager = CortexOSSyncManager()
    
    # Test sync manager startup
    if sync_manager.start():
        print("✅ Sync Manager started successfully")
    
    # Register test components
    test_components = ["neuroengine", "context_engine", "resonance_field"]
    for component in test_components:
        sync_manager.register_component(component)
        print(f"✅ Registered component: {component}")
    
    # Test synchronization
    print("🔄 Testing synchronization...")
    time.sleep(2)  # Let sync run
    
    # Test sync status
    status = sync_manager.get_sync_status()
    print(f"✅ Sync status: {status}")
    
    # Test force sync
    if sync_manager.force_sync():
        print("✅ Force sync completed")
    
    # Test component unregistration
    sync_manager.unregister_component("neuroengine")
    print("✅ Unregistered test component")
    
    # Shutdown
    sync_manager.stop()
    print("✅ Sync Manager stopped")
    
    print("🔄 Sync Manager test complete!")

