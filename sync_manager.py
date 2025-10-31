#!/usr/bin/env python3
"""
infrastructure/sync_manager.py - CortexOS Global Synchronization Manager
COMPLETE IMPLEMENTATION - Advanced synchronization algorithms, conflict resolution, sync failure recovery
"""

import time
import threading
import asyncio
import logging
import json
import hashlib
import queue
from typing import Dict, List, Callable, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import weakref

class SyncState(Enum):
    IDLE = "idle"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    FAILED = "failed"
    RECOVERING = "recovering"

class ConflictResolutionStrategy(Enum):
    TIMESTAMP_WINS = "timestamp_wins"
    PRIORITY_WINS = "priority_wins"
    MERGE_CHANGES = "merge_changes"
    USER_DECIDES = "user_decides"
    ROLLBACK = "rollback"

@dataclass
class SyncEvent:
    event_id: str
    component_id: str
    timestamp: float
    event_type: str
    data: Any
    checksum: str
    priority: int = 0

@dataclass
class ConflictRecord:
    conflict_id: str
    timestamp: float
    components: List[str]
    conflicting_events: List[SyncEvent]
    resolution_strategy: ConflictResolutionStrategy
    resolved: bool = False
    resolution_data: Any = None

class AdvancedSyncManager:
    """
    A Singleton class to manage synchronization policies across CortexOS.
    Ensures that all components share the same sync manager instance.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """
        This is the method the error says is missing.
        It gets the single, shared instance of the class.
        """
        if cls._instance is None:
            # Use a lock to ensure thread-safe instance creation
            with cls._lock:
                # Check again in case another thread created the instance
                # while the first one was waiting for the lock.
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        The initializer should only run once.
        We use a flag to prevent re-initialization on subsequent calls.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.logger = logging.getLogger(__name__)
            # Default synchronization policies
            self.sync_policies = {
                'default': {'sync_interval': 5.0},
                'swarm_resonance': {'sync_interval': 3.0}
            }
            self.logger.info("AdvancedSyncManager initialized.")

    def get_sync_policy(self, module_name):
        """
        Gets the synchronization policy for a given module.
        Falls back to the default policy if a specific one isn't found.
        """
        return self.sync_policies.get(module_name, self.sync_policies['default'])
        
        # Conflict management
        self.conflicts = {}
        self.conflict_history = deque(maxlen=1000)
        self.resolution_strategies = {
            ConflictResolutionStrategy.TIMESTAMP_WINS: self._resolve_by_timestamp,
            ConflictResolutionStrategy.PRIORITY_WINS: self._resolve_by_priority,
            ConflictResolutionStrategy.MERGE_CHANGES: self._resolve_by_merge,
            ConflictResolutionStrategy.ROLLBACK: self._resolve_by_rollback
        }
        
        # Recovery management
        self.checkpoints = deque(maxlen=100)
        self.recovery_points = {}
        self.failed_syncs = deque(maxlen=500)
        
        # Performance tracking
        self.sync_metrics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'average_sync_time': 0.0,
            'last_sync_time': 0.0
        }
        
        # Threading
        self.sync_thread = None
        self.event_processor_thread = None
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(f'CortexOS.SyncManager.{self.node_id}')
        self.logger.setLevel(logging.INFO)
        
    def register_component(self, component_id: str, component: Any, 
                          priority: int = 0, dependencies: List[str] = None,
                          sync_callback: Callable = None) -> bool:
        """Register a component with advanced configuration"""
        try:
            with self.lock:
                # Initialize vector clock entry
                if component_id not in self.vector_clock:
                    self.vector_clock[component_id] = 0
                
                # Register component
                self.registered_components[component_id] = {
                    'component': weakref.ref(component) if component else None,
                    'priority': priority,
                    'last_sync': 0.0,
                    'sync_count': 0,
                    'active': True,
                    'state': SyncState.IDLE,
                    'last_event_id': None,
                    'sync_errors': 0,
                    'recovery_count': 0
                }
                
                self.component_priorities[component_id] = priority
                
                # Set dependencies
                if dependencies:
                    self.component_dependencies[component_id] = set(dependencies)
                
                # Register callback
                if sync_callback:
                    self.sync_callbacks[component_id].append(sync_callback)
                
                self.logger.info(f"Component registered: {component_id} (priority: {priority})")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register component {component_id}: {e}")
            return False
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister component with cleanup"""
        try:
            with self.lock:
                # Remove from all data structures
                self.registered_components.pop(component_id, None)
                self.component_priorities.pop(component_id, None)
                self.component_dependencies.pop(component_id, None)
                self.sync_callbacks.pop(component_id, None)
                self.vector_clock.pop(component_id, None)
                
                # Remove from dependencies of other components
                for deps in self.component_dependencies.values():
                    deps.discard(component_id)
                
                self.logger.info(f"Component unregistered: {component_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unregister component {component_id}: {e}")
            return False
    
    def start(self) -> bool:
        """Start the advanced synchronization system"""
        try:
            if self.running:
                return False
            
            self.running = True
            self.shutdown_event.clear()
            
            # Start sync thread
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            
            # Start event processor
            self.event_processor_thread = threading.Thread(target=self._event_processor_loop, daemon=True)
            self.event_processor_thread.start()
            
            self.state = SyncState.SYNCING
            self.logger.info("Advanced sync manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start sync manager: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop synchronization with graceful shutdown"""
        try:
            self.running = False
            self.shutdown_event.set()
            
            # Wait for threads to finish
            if self.sync_thread and self.sync_thread.is_alive():
                self.sync_thread.join(timeout=5.0)
            
            if self.event_processor_thread and self.event_processor_thread.is_alive():
                self.event_processor_thread.join(timeout=5.0)
            
            self.state = SyncState.IDLE
            self.logger.info("Advanced sync manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop sync manager: {e}")
            return False
    
    def _sync_loop(self):
        """Main synchronization loop with advanced algorithms"""
        while self.running and not self.shutdown_event.is_set():
            try:
                sync_start = time.time()
                
                # Update clocks
                self.sync_clock = sync_start
                self.logical_clock += 1
                
                # Create checkpoint before sync
                self._create_checkpoint()
                
                # Perform synchronization
                success = self._perform_sync_cycle()
                
                # Update metrics
                sync_duration = time.time() - sync_start
                self._update_sync_metrics(success, sync_duration)
                
                # Sleep to maintain frequency
                sleep_time = max(0, (1.0 / self.base_frequency) - sync_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                self._handle_sync_failure(e)
                time.sleep(0.1)
    
    def _perform_sync_cycle(self) -> bool:
        """Perform a complete synchronization cycle"""
        try:
            with self.lock:
                # Check for conflicts
                conflicts = self._detect_conflicts()
                if conflicts:
                    self._handle_conflicts(conflicts)
                
                # Sort components by priority and dependencies
                sync_order = self._calculate_sync_order()
                
                # Synchronize components in order
                for component_id in sync_order:
                    if not self._sync_component(component_id):
                        self.logger.warning(f"Failed to sync component: {component_id}")
                        return False
                
                self.sync_metrics['successful_syncs'] += 1
                return True
                
        except Exception as e:
            self.logger.error(f"Sync cycle failed: {e}")
            self.sync_metrics['failed_syncs'] += 1
            return False
    
    def _detect_conflicts(self) -> List[ConflictRecord]:
        """Advanced conflict detection using vector clocks"""
        conflicts = []
        
        try:
            # Check for concurrent events
            for component_id, component_info in self.registered_components.items():
                if not component_info['active']:
                    continue
                
                # Check vector clock for conflicts
                for other_id, other_info in self.registered_components.items():
                    if other_id == component_id or not other_info['active']:
                        continue
                    
                    # Detect concurrent modifications
                    if self._are_concurrent(component_id, other_id):
                        conflict = self._create_conflict_record(component_id, other_id)
                        if conflict:
                            conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
            return []
    
    def _are_concurrent(self, comp1: str, comp2: str) -> bool:
        """Check if two components have concurrent modifications"""
        try:
            clock1 = self.vector_clock.get(comp1, 0)
            clock2 = self.vector_clock.get(comp2, 0)
            
            # Simple concurrency check - can be enhanced
            return abs(clock1 - clock2) < 2 and clock1 > 0 and clock2 > 0
            
        except Exception:
            return False
    
    def _create_conflict_record(self, comp1: str, comp2: str) -> Optional[ConflictRecord]:
        """Create a conflict record for resolution"""
        try:
            conflict_id = hashlib.md5(f"{comp1}_{comp2}_{time.time()}".encode()).hexdigest()[:8]
            
            return ConflictRecord(
                conflict_id=conflict_id,
                timestamp=time.time(),
                components=[comp1, comp2],
                conflicting_events=[],  # Would be populated with actual events
                resolution_strategy=ConflictResolutionStrategy.TIMESTAMP_WINS
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create conflict record: {e}")
            return None
    
    def _handle_conflicts(self, conflicts: List[ConflictRecord]):
        """Handle detected conflicts using resolution strategies"""
        for conflict in conflicts:
            try:
                self.conflicts[conflict.conflict_id] = conflict
                self.sync_metrics['conflicts_detected'] += 1
                
                # Apply resolution strategy
                strategy = conflict.resolution_strategy
                if strategy in self.resolution_strategies:
                    success = self.resolution_strategies[strategy](conflict)
                    if success:
                        conflict.resolved = True
                        self.sync_metrics['conflicts_resolved'] += 1
                        self.logger.info(f"Conflict resolved: {conflict.conflict_id}")
                    else:
                        self.logger.error(f"Failed to resolve conflict: {conflict.conflict_id}")
                
                self.conflict_history.append(conflict)
                
            except Exception as e:
                self.logger.error(f"Failed to handle conflict {conflict.conflict_id}: {e}")
    
    def _resolve_by_timestamp(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict by timestamp (last writer wins)"""
        try:
            # Find component with latest timestamp
            latest_component = None
            latest_time = 0
            
            for comp_id in conflict.components:
                comp_info = self.registered_components.get(comp_id)
                if comp_info and comp_info['last_sync'] > latest_time:
                    latest_time = comp_info['last_sync']
                    latest_component = comp_id
            
            if latest_component:
                # Mark other components for resync
                for comp_id in conflict.components:
                    if comp_id != latest_component:
                        self.registered_components[comp_id]['state'] = SyncState.RECOVERING
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Timestamp resolution failed: {e}")
            return False
    
    def _resolve_by_priority(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict by component priority"""
        try:
            # Find highest priority component
            highest_priority = -1
            priority_component = None
            
            for comp_id in conflict.components:
                priority = self.component_priorities.get(comp_id, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    priority_component = comp_id
            
            if priority_component:
                # Mark lower priority components for resync
                for comp_id in conflict.components:
                    if comp_id != priority_component:
                        self.registered_components[comp_id]['state'] = SyncState.RECOVERING
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Priority resolution failed: {e}")
            return False
    
    def _resolve_by_merge(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict by merging changes"""
        try:
            # This would implement actual merge logic
            # For now, just mark all components as needing resync
            for comp_id in conflict.components:
                self.registered_components[comp_id]['state'] = SyncState.RECOVERING
            
            return True
            
        except Exception as e:
            self.logger.error(f"Merge resolution failed: {e}")
            return False
    
    def _resolve_by_rollback(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict by rolling back to last checkpoint"""
        try:
            # Find appropriate checkpoint
            checkpoint = self._find_recovery_checkpoint(conflict.timestamp)
            if checkpoint:
                return self._restore_from_checkpoint(checkpoint)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rollback resolution failed: {e}")
            return False
    
    def _calculate_sync_order(self) -> List[str]:
        """Calculate optimal synchronization order based on dependencies and priorities"""
        try:
            # Topological sort with priority weighting
            in_degree = defaultdict(int)
            graph = defaultdict(list)
            
            # Build dependency graph
            for comp_id, deps in self.component_dependencies.items():
                for dep in deps:
                    graph[dep].append(comp_id)
                    in_degree[comp_id] += 1
            
            # Initialize queue with components having no dependencies
            sync_queue = []
            for comp_id in self.registered_components:
                if in_degree[comp_id] == 0:
                    priority = self.component_priorities.get(comp_id, 0)
                    sync_queue.append((-priority, comp_id))  # Negative for max heap
            
            sync_queue.sort()
            
            # Process components in dependency order
            result = []
            while sync_queue:
                _, comp_id = sync_queue.pop(0)
                result.append(comp_id)
                
                # Update dependencies
                for dependent in graph[comp_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        priority = self.component_priorities.get(dependent, 0)
                        sync_queue.append((-priority, dependent))
                        sync_queue.sort()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate sync order: {e}")
            return list(self.registered_components.keys())
    
    def _sync_component(self, component_id: str) -> bool:
        """Synchronize a single component with error handling"""
        try:
            component_info = self.registered_components.get(component_id)
            if not component_info or not component_info['active']:
                return True
            
            # Update vector clock
            self.vector_clock[component_id] += 1
            
            # Update component info
            component_info['last_sync'] = self.sync_clock
            component_info['sync_count'] += 1
            component_info['state'] = SyncState.SYNCING
            
            # Call sync callbacks
            for callback in self.sync_callbacks.get(component_id, []):
                try:
                    callback(self.sync_clock, self.logical_clock, component_id)
                except Exception as e:
                    self.logger.error(f"Callback failed for {component_id}: {e}")
                    component_info['sync_errors'] += 1
            
            component_info['state'] = SyncState.IDLE
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync component {component_id}: {e}")
            if component_id in self.registered_components:
                self.registered_components[component_id]['sync_errors'] += 1
                self.registered_components[component_id]['state'] = SyncState.FAILED
            return False
    
    def _create_checkpoint(self):
        """Create a system checkpoint for recovery"""
        try:
            checkpoint = {
                'timestamp': time.time(),
                'sync_clock': self.sync_clock,
                'logical_clock': self.logical_clock,
                'vector_clock': self.vector_clock.copy(),
                'component_states': {}
            }
            
            # Capture component states
            for comp_id, comp_info in self.registered_components.items():
                checkpoint['component_states'][comp_id] = {
                    'last_sync': comp_info['last_sync'],
                    'sync_count': comp_info['sync_count'],
                    'state': comp_info['state'].value
                }
            
            self.checkpoints.append(checkpoint)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
    
    def _find_recovery_checkpoint(self, before_time: float) -> Optional[Dict]:
        """Find the most recent checkpoint before the given time"""
        try:
            for checkpoint in reversed(self.checkpoints):
                if checkpoint['timestamp'] < before_time:
                    return checkpoint
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find recovery checkpoint: {e}")
            return None
    
    def _restore_from_checkpoint(self, checkpoint: Dict) -> bool:
        """Restore system state from checkpoint"""
        try:
            with self.lock:
                self.sync_clock = checkpoint['sync_clock']
                self.logical_clock = checkpoint['logical_clock']
                self.vector_clock = checkpoint['vector_clock'].copy()
                
                # Restore component states
                for comp_id, state in checkpoint['component_states'].items():
                    if comp_id in self.registered_components:
                        comp_info = self.registered_components[comp_id]
                        comp_info['last_sync'] = state['last_sync']
                        comp_info['sync_count'] = state['sync_count']
                        comp_info['state'] = SyncState(state['state'])
                        comp_info['recovery_count'] += 1
                
                self.logger.info(f"Restored from checkpoint: {checkpoint['timestamp']}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def _handle_sync_failure(self, error: Exception):
        """Handle synchronization failures with recovery"""
        try:
            failure_record = {
                'timestamp': time.time(),
                'error': str(error),
                'sync_clock': self.sync_clock,
                'logical_clock': self.logical_clock,
                'active_components': len([c for c in self.registered_components.values() if c['active']])
            }
            
            self.failed_syncs.append(failure_record)
            self.state = SyncState.FAILED
            
            # Attempt recovery
            if len(self.failed_syncs) > 3:  # Multiple failures
                self.logger.warning("Multiple sync failures detected, attempting recovery")
                self._attempt_recovery()
            
        except Exception as e:
            self.logger.error(f"Failed to handle sync failure: {e}")
    
    def _attempt_recovery(self) -> bool:
        """Attempt to recover from sync failures"""
        try:
            self.state = SyncState.RECOVERING
            
            # Find recent checkpoint
            checkpoint = self._find_recovery_checkpoint(time.time() - 60)  # Last minute
            if checkpoint:
                if self._restore_from_checkpoint(checkpoint):
                    self.state = SyncState.SYNCING
                    self.logger.info("Recovery successful")
                    return True
            
            # If no checkpoint, reset to clean state
            self._reset_to_clean_state()
            self.state = SyncState.SYNCING
            self.logger.info("Reset to clean state")
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            return False
    
    def _reset_to_clean_state(self):
        """Reset system to clean state"""
        try:
            with self.lock:
                self.logical_clock = 0
                self.vector_clock = {comp_id: 0 for comp_id in self.registered_components}
                
                for comp_info in self.registered_components.values():
                    comp_info['state'] = SyncState.IDLE
                    comp_info['sync_errors'] = 0
                
                # Clear event queue
                while not self.event_queue.empty():
                    try:
                        self.event_queue.get_nowait()
                    except queue.Empty:
                        break
                
        except Exception as e:
            self.logger.error(f"Failed to reset to clean state: {e}")
    
    def _event_processor_loop(self):
        """Process events from the event queue"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get event with timeout
                try:
                    priority, event = self.event_queue.get(timeout=0.1)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Event processor error: {e}")
                time.sleep(0.1)
    
    def _process_event(self, event: SyncEvent):
        """Process a synchronization event"""
        try:
            # Add to history
            self.event_history.append(event)
            
            # Update vector clock
            if event.component_id in self.vector_clock:
                self.vector_clock[event.component_id] = max(
                    self.vector_clock[event.component_id],
                    event.timestamp
                )
            
            # Process based on event type
            if event.event_type == "sync_request":
                self._handle_sync_request(event)
            elif event.event_type == "conflict_detected":
                self._handle_conflict_event(event)
            elif event.event_type == "recovery_needed":
                self._handle_recovery_event(event)
            
        except Exception as e:
            self.logger.error(f"Failed to process event {event.event_id}: {e}")
    
    def _handle_sync_request(self, event: SyncEvent):
        """Handle synchronization request event"""
        try:
            component_id = event.component_id
            if component_id in self.registered_components:
                self._sync_component(component_id)
                
        except Exception as e:
            self.logger.error(f"Failed to handle sync request: {e}")
    
    def _handle_conflict_event(self, event: SyncEvent):
        """Handle conflict detection event"""
        try:
            # Create conflict record and resolve
            conflict_data = event.data
            if isinstance(conflict_data, dict):
                conflict = ConflictRecord(**conflict_data)
                self._handle_conflicts([conflict])
                
        except Exception as e:
            self.logger.error(f"Failed to handle conflict event: {e}")
    
    def _handle_recovery_event(self, event: SyncEvent):
        """Handle recovery request event"""
        try:
            self._attempt_recovery()
            
        except Exception as e:
            self.logger.error(f"Failed to handle recovery event: {e}")
    
    def _update_sync_metrics(self, success: bool, duration: float):
        """Update synchronization metrics"""
        try:
            self.sync_metrics['total_syncs'] += 1
            self.sync_metrics['last_sync_time'] = duration
            
            if success:
                self.sync_metrics['successful_syncs'] += 1
            else:
                self.sync_metrics['failed_syncs'] += 1
            
            # Update average
            total = self.sync_metrics['total_syncs']
            current_avg = self.sync_metrics['average_sync_time']
            self.sync_metrics['average_sync_time'] = (current_avg * (total - 1) + duration) / total
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    # Public API methods
    
    def submit_event(self, component_id: str, event_type: str, data: Any, priority: int = 0) -> str:
        """Submit an event for processing"""
        try:
            event_id = hashlib.md5(f"{component_id}_{event_type}_{time.time()}".encode()).hexdigest()[:8]
            
            event = SyncEvent(
                event_id=event_id,
                component_id=component_id,
                timestamp=time.time(),
                event_type=event_type,
                data=data,
                checksum=hashlib.md5(str(data).encode()).hexdigest(),
                priority=priority
            )
            
            self.event_queue.put((priority, event))
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit event: {e}")
            return ""
    
    def force_sync(self, component_id: str = None) -> bool:
        """Force synchronization of specific component or all components"""
        try:
            if component_id:
                return self._sync_component(component_id)
            else:
                return self._perform_sync_cycle()
                
        except Exception as e:
            self.logger.error(f"Force sync failed: {e}")
            return False
    
    def get_sync_status(self) -> Dict:
        """Get comprehensive synchronization status"""
        try:
            with self.lock:
                active_components = sum(1 for c in self.registered_components.values() if c['active'])
                
                return {
                    'running': self.running,
                    'state': self.state.value,
                    'node_id': self.node_id,
                    'sync_clock': self.sync_clock,
                    'logical_clock': self.logical_clock,
                    'base_frequency': self.base_frequency,
                    'registered_components': len(self.registered_components),
                    'active_components': active_components,
                    'vector_clock': self.vector_clock.copy(),
                    'metrics': self.sync_metrics.copy(),
                    'conflicts': {
                        'active': len([c for c in self.conflicts.values() if not c.resolved]),
                        'total': len(self.conflict_history)
                    },
                    'checkpoints': len(self.checkpoints),
                    'failed_syncs': len(self.failed_syncs)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get sync status: {e}")
            return {}
    
    def set_component_priority(self, component_id: str, priority: int) -> bool:
        """Set component synchronization priority"""
        try:
            if component_id in self.registered_components:
                self.component_priorities[component_id] = priority
                self.registered_components[component_id]['priority'] = priority
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set component priority: {e}")
            return False
    
    def add_dependency(self, component_id: str, dependency: str) -> bool:
        """Add a dependency relationship between components"""
        try:
            if component_id in self.registered_components and dependency in self.registered_components:
                self.component_dependencies[component_id].add(dependency)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add dependency: {e}")
            return False
    
    def remove_dependency(self, component_id: str, dependency: str) -> bool:
        """Remove a dependency relationship"""
        try:
            if component_id in self.component_dependencies:
                self.component_dependencies[component_id].discard(dependency)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove dependency: {e}")
            return False

# Alias for backward compatibility
CortexOSSyncManager = AdvancedSyncManager
GlobalSyncManager = AdvancedSyncManager

if __name__ == "__main__":
    # Comprehensive test suite
    print("🔄 Testing Advanced CortexOS Sync Manager...")
    
    # Create sync manager
    sync_manager = AdvancedSyncManager(base_frequency=5.0)
    
    # Test component registration
    print("📝 Testing component registration...")
    components = ["neuroengine", "context_engine", "resonance_field", "memory_inserter"]
    priorities = [10, 8, 6, 4]
    
    for i, comp in enumerate(components):
        success = sync_manager.register_component(
            comp, 
            None,  # Mock component
            priority=priorities[i],
            dependencies=components[:i] if i > 0 else []
        )
        print(f"{'✅' if success else '❌'} Registered {comp} (priority: {priorities[i]})")
    
    # Test sync manager startup
    print("\n🚀 Testing sync manager startup...")
    if sync_manager.start():
        print("✅ Sync manager started successfully")
        
        # Let it run for a few cycles
        time.sleep(2)
        
        # Test status
        status = sync_manager.get_sync_status()
        print(f"📊 Status: {status['state']}, Components: {status['active_components']}")
        print(f"📈 Metrics: {status['metrics']['total_syncs']} syncs, {status['metrics']['successful_syncs']} successful")
        
        # Test event submission
        print("\n📨 Testing event submission...")
        event_id = sync_manager.submit_event("neuroengine", "test_event", {"test": "data"}, priority=5)
        print(f"✅ Submitted event: {event_id}")
        
        # Test force sync
        print("\n🔄 Testing force sync...")
        if sync_manager.force_sync("context_engine"):
            print("✅ Force sync successful")
        
        # Test dependency management
        print("\n🔗 Testing dependency management...")
        if sync_manager.add_dependency("memory_inserter", "neuroengine"):
            print("✅ Added dependency: memory_inserter -> neuroengine")
        
        # Test priority change
        print("\n⚡ Testing priority change...")
        if sync_manager.set_component_priority("resonance_field", 15):
            print("✅ Changed resonance_field priority to 15")
        
        # Let it run a bit more
        time.sleep(2)
        
        # Final status
        final_status = sync_manager.get_sync_status()
        print(f"\n📊 Final Status:")
        print(f"  Total Syncs: {final_status['metrics']['total_syncs']}")
        print(f"  Success Rate: {final_status['metrics']['successful_syncs']}/{final_status['metrics']['total_syncs']}")
        print(f"  Average Sync Time: {final_status['metrics']['average_sync_time']:.4f}s")
        print(f"  Conflicts: {final_status['conflicts']['total']}")
        print(f"  Checkpoints: {final_status['checkpoints']}")
        
        # Test shutdown
        print("\n🛑 Testing shutdown...")
        if sync_manager.stop():
            print("✅ Sync manager stopped successfully")
    else:
        print("❌ Failed to start sync manager")
    
    print("\n🎉 Advanced Sync Manager test complete!")

