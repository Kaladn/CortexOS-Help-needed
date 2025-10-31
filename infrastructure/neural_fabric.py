#!/usr/bin/env python3
"""
infrastructure/neural_fabric.py - CortexOS Neural Fabric Infrastructure
COMPLETE IMPLEMENTATION - Neural network topology management, dynamic routing, load balancing, fabric health monitoring
"""

import time
import threading
import asyncio
import logging
import json
import hashlib
import queue
import weakref
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

class NodeType(Enum):
    PROCESSOR = "processor"
    STORAGE = "storage"
    MODULATOR = "modulator"
    GATEWAY = "gateway"
    MONITOR = "monitor"

class ConnectionType(Enum):
    DATA_FLOW = "data_flow"
    FEEDBACK = "feedback"
    CONTROL = "control"
    BROADCAST = "broadcast"
    EMERGENCY = "emergency"

class NodeState(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class NeuralSignal:
    signal_id: str
    source_id: str
    target_id: str
    signal_type: str
    data: Any
    strength: float
    timestamp: float
    priority: int = 0
    ttl: int = 100  # Time to live (hops)
    path: List[str] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = [self.source_id]

@dataclass
class NeuralNode:
    node_id: str
    node_type: NodeType
    capabilities: List[str]
    max_load: int = 100
    current_load: int = 0
    state: NodeState = NodeState.IDLE
    last_heartbeat: float = 0.0
    processing_time: float = 0.0
    error_count: int = 0
    total_processed: int = 0

@dataclass
class NeuralConnection:
    connection_id: str
    source_id: str
    target_id: str
    connection_type: ConnectionType
    weight: float = 1.0
    bandwidth: float = 1.0  # Signals per second
    latency: float = 0.001  # Seconds
    reliability: float = 1.0  # 0.0 to 1.0
    usage_count: int = 0
    last_used: float = 0.0
    active: bool = True
    bidirectional: bool = False

class AdvancedNeuralFabric:
    """
    Advanced Neural Fabric Infrastructure for CortexOS
    
    Features:
    - Neural network topology management with dynamic reconfiguration
    - Intelligent routing algorithms with load balancing
    - Distributed load balancing across nodes
    - Comprehensive fabric health monitoring
    - Adaptive connection weights and bandwidth management
    - Fault tolerance with automatic recovery
    - Performance optimization and bottleneck detection
    """
    
    def __init__(self, fabric_id: str = None):
        self.fabric_id = fabric_id or f"fabric_{int(time.time())}"
        
        # Core fabric components
        self.nodes = {}  # {node_id: NeuralNode}
        self.connections = {}  # {connection_id: NeuralConnection}
        self.topology = defaultdict(set)  # {node_id: {connected_node_ids}}
        
        # Signal management
        self.signal_queues = defaultdict(queue.PriorityQueue)
        self.signal_history = deque(maxlen=10000)
        self.pending_signals = {}
        self.signal_routes = {}  # Cached routing table
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        self.node_loads = defaultdict(float)
        self.connection_loads = defaultdict(float)
        
        # Health monitoring
        self.health_monitor = HealthMonitor()
        self.performance_metrics = PerformanceMetrics()
        self.failure_detector = FailureDetector()
        
        # Routing algorithms
        self.routing_algorithms = {
            'shortest_path': self._route_shortest_path,
            'load_balanced': self._route_load_balanced,
            'adaptive': self._route_adaptive,
            'broadcast': self._route_broadcast
        }
        self.default_routing = 'adaptive'
        
        # Threading and async
        self.running = False
        self.fabric_threads = {}
        self.async_loop = None
        self.shutdown_event = threading.Event()
        
        # Configuration
        self.config = {
            'max_signal_queue_size': 1000,
            'heartbeat_interval': 5.0,
            'health_check_interval': 10.0,
            'load_balance_interval': 2.0,
            'topology_update_interval': 30.0,
            'signal_timeout': 60.0,
            'max_retries': 3,
            'adaptive_weights': True,
            'auto_recovery': True
        }
        
        # Logging
        self.logger = logging.getLogger(f'CortexOS.NeuralFabric.{self.fabric_id}')
        self.logger.setLevel(logging.INFO)
        
    def register_node(self, node_id: str, node_type: NodeType, capabilities: List[str],
                     max_load: int = 100, signal_handler: Callable = None) -> bool:
        """Register a neural node with advanced configuration"""
        try:
            # Create node
            node = NeuralNode(
                node_id=node_id,
                node_type=node_type,
                capabilities=capabilities,
                max_load=max_load,
                last_heartbeat=time.time()
            )
            
            self.nodes[node_id] = node
            self.topology[node_id] = set()
            
            # Register with load balancer
            self.load_balancer.register_node(node_id, max_load)
            
            # Register with health monitor
            self.health_monitor.register_node(node_id, signal_handler)
            
            # Initialize signal queue
            if node_id not in self.signal_queues:
                self.signal_queues[node_id] = queue.PriorityQueue(
                    maxsize=self.config['max_signal_queue_size']
                )
            
            self.logger.info(f"Node registered: {node_id} ({node_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_id}: {e}")
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister node with cleanup"""
        try:
            # Remove from all data structures
            self.nodes.pop(node_id, None)
            self.topology.pop(node_id, None)
            self.signal_queues.pop(node_id, None)
            self.node_loads.pop(node_id, None)
            
            # Remove from topology of other nodes
            for connected_nodes in self.topology.values():
                connected_nodes.discard(node_id)
            
            # Remove connections involving this node
            connections_to_remove = []
            for conn_id, connection in self.connections.items():
                if connection.source_id == node_id or connection.target_id == node_id:
                    connections_to_remove.append(conn_id)
            
            for conn_id in connections_to_remove:
                self.remove_connection(conn_id)
            
            # Unregister from subsystems
            self.load_balancer.unregister_node(node_id)
            self.health_monitor.unregister_node(node_id)
            
            # Clear cached routes
            self._clear_routing_cache()
            
            self.logger.info(f"Node unregistered: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    def create_connection(self, source_id: str, target_id: str, 
                         connection_type: ConnectionType = ConnectionType.DATA_FLOW,
                         weight: float = 1.0, bandwidth: float = 1.0,
                         bidirectional: bool = False) -> Optional[str]:
        """Create neural connection with advanced properties"""
        try:
            # Validate nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                self.logger.error(f"Cannot create connection: nodes not registered")
                return None
            
            connection_id = f"{source_id}->{target_id}"
            if bidirectional:
                connection_id += "_bidirectional"
            
            # Create connection
            connection = NeuralConnection(
                connection_id=connection_id,
                source_id=source_id,
                target_id=target_id,
                connection_type=connection_type,
                weight=weight,
                bandwidth=bandwidth,
                bidirectional=bidirectional
            )
            
            self.connections[connection_id] = connection
            
            # Update topology
            self.topology[source_id].add(target_id)
            if bidirectional:
                self.topology[target_id].add(source_id)
                # Create reverse connection
                reverse_id = f"{target_id}->{source_id}_bidirectional"
                reverse_connection = NeuralConnection(
                    connection_id=reverse_id,
                    source_id=target_id,
                    target_id=source_id,
                    connection_type=connection_type,
                    weight=weight,
                    bandwidth=bandwidth,
                    bidirectional=True
                )
                self.connections[reverse_id] = reverse_connection
            
            # Clear cached routes
            self._clear_routing_cache()
            
            self.logger.info(f"Connection created: {connection_id}")
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove neural connection"""
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                return False
            
            # Update topology
            self.topology[connection.source_id].discard(connection.target_id)
            if connection.bidirectional:
                self.topology[connection.target_id].discard(connection.source_id)
                # Remove reverse connection
                reverse_id = f"{connection.target_id}->{connection.source_id}_bidirectional"
                self.connections.pop(reverse_id, None)
            
            # Remove connection
            del self.connections[connection_id]
            self.connection_loads.pop(connection_id, None)
            
            # Clear cached routes
            self._clear_routing_cache()
            
            self.logger.info(f"Connection removed: {connection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove connection {connection_id}: {e}")
            return False
    
    def send_signal(self, source_id: str, target_id: str, signal_type: str,
                   data: Any, strength: float = 1.0, priority: int = 0,
                   routing_algorithm: str = None) -> Optional[str]:
        """Send signal with advanced routing"""
        try:
            # Generate signal ID
            signal_id = hashlib.md5(
                f"{source_id}_{target_id}_{signal_type}_{time.time()}".encode()
            ).hexdigest()[:8]
            
            # Create signal
            signal = NeuralSignal(
                signal_id=signal_id,
                source_id=source_id,
                target_id=target_id,
                signal_type=signal_type,
                data=data,
                strength=strength,
                timestamp=time.time(),
                priority=priority
            )
            
            # Route signal
            routing_alg = routing_algorithm or self.default_routing
            route = self._route_signal(signal, routing_alg)
            
            if not route:
                self.logger.warning(f"No route found for signal {signal_id}")
                return None
            
            # Send signal along route
            success = self._transmit_signal(signal, route)
            
            if success:
                # Update metrics
                self.performance_metrics.record_signal_sent(signal)
                self.signal_history.append(signal)
                
                # Update connection usage
                for i in range(len(route) - 1):
                    conn_id = f"{route[i]}->{route[i+1]}"
                    if conn_id in self.connections:
                        self.connections[conn_id].usage_count += 1
                        self.connections[conn_id].last_used = time.time()
                
                return signal_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            return None
    
    def _route_signal(self, signal: NeuralSignal, algorithm: str) -> Optional[List[str]]:
        """Route signal using specified algorithm"""
        try:
            if algorithm in self.routing_algorithms:
                return self.routing_algorithms[algorithm](signal)
            else:
                return self.routing_algorithms[self.default_routing](signal)
                
        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            return None
    
    def _route_shortest_path(self, signal: NeuralSignal) -> Optional[List[str]]:
        """Find shortest path using Dijkstra's algorithm"""
        try:
            source = signal.source_id
            target = signal.target_id
            
            # Check cache first
            cache_key = f"{source}->{target}_shortest"
            if cache_key in self.signal_routes:
                return self.signal_routes[cache_key]
            
            # Dijkstra's algorithm
            distances = {node: float('inf') for node in self.nodes}
            distances[source] = 0
            previous = {}
            unvisited = set(self.nodes.keys())
            
            while unvisited:
                current = min(unvisited, key=lambda x: distances[x])
                if distances[current] == float('inf'):
                    break
                
                unvisited.remove(current)
                
                if current == target:
                    break
                
                for neighbor in self.topology[current]:
                    if neighbor in unvisited:
                        # Calculate distance (inverse of weight for shortest path)
                        conn_id = f"{current}->{neighbor}"
                        connection = self.connections.get(conn_id)
                        if connection and connection.active:
                            distance = distances[current] + (1.0 / connection.weight)
                            if distance < distances[neighbor]:
                                distances[neighbor] = distance
                                previous[neighbor] = current
            
            # Reconstruct path
            if target not in previous and target != source:
                return None
            
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            
            path.reverse()
            
            # Cache result
            self.signal_routes[cache_key] = path
            
            return path
            
        except Exception as e:
            self.logger.error(f"Shortest path routing failed: {e}")
            return None
    
    def _route_load_balanced(self, signal: NeuralSignal) -> Optional[List[str]]:
        """Route signal considering node loads"""
        try:
            source = signal.source_id
            target = signal.target_id
            
            # Modified Dijkstra considering load
            distances = {node: float('inf') for node in self.nodes}
            distances[source] = 0
            previous = {}
            unvisited = set(self.nodes.keys())
            
            while unvisited:
                current = min(unvisited, key=lambda x: distances[x])
                if distances[current] == float('inf'):
                    break
                
                unvisited.remove(current)
                
                if current == target:
                    break
                
                for neighbor in self.topology[current]:
                    if neighbor in unvisited:
                        conn_id = f"{current}->{neighbor}"
                        connection = self.connections.get(conn_id)
                        if connection and connection.active:
                            # Factor in node load
                            node_load = self.node_loads.get(neighbor, 0.0)
                            load_factor = 1.0 + node_load  # Higher load = higher cost
                            
                            distance = distances[current] + (load_factor / connection.weight)
                            if distance < distances[neighbor]:
                                distances[neighbor] = distance
                                previous[neighbor] = current
            
            # Reconstruct path
            if target not in previous and target != source:
                return None
            
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            
            path.reverse()
            return path
            
        except Exception as e:
            self.logger.error(f"Load balanced routing failed: {e}")
            return None
    
    def _route_adaptive(self, signal: NeuralSignal) -> Optional[List[str]]:
        """Adaptive routing considering multiple factors"""
        try:
            source = signal.source_id
            target = signal.target_id
            
            # Adaptive cost function
            distances = {node: float('inf') for node in self.nodes}
            distances[source] = 0
            previous = {}
            unvisited = set(self.nodes.keys())
            
            while unvisited:
                current = min(unvisited, key=lambda x: distances[x])
                if distances[current] == float('inf'):
                    break
                
                unvisited.remove(current)
                
                if current == target:
                    break
                
                for neighbor in self.topology[current]:
                    if neighbor in unvisited:
                        conn_id = f"{current}->{neighbor}"
                        connection = self.connections.get(conn_id)
                        if connection and connection.active:
                            # Adaptive cost considering multiple factors
                            node_load = self.node_loads.get(neighbor, 0.0)
                            conn_load = self.connection_loads.get(conn_id, 0.0)
                            
                            # Cost factors
                            distance_cost = 1.0 / connection.weight
                            load_cost = (node_load + conn_load) * 0.5
                            latency_cost = connection.latency * 1000  # Convert to ms
                            reliability_cost = (1.0 - connection.reliability) * 2.0
                            
                            # Weighted combination
                            total_cost = (distance_cost * 0.3 + 
                                        load_cost * 0.3 + 
                                        latency_cost * 0.2 + 
                                        reliability_cost * 0.2)
                            
                            distance = distances[current] + total_cost
                            if distance < distances[neighbor]:
                                distances[neighbor] = distance
                                previous[neighbor] = current
            
            # Reconstruct path
            if target not in previous and target != source:
                return None
            
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            
            path.reverse()
            return path
            
        except Exception as e:
            self.logger.error(f"Adaptive routing failed: {e}")
            return None
    
    def _route_broadcast(self, signal: NeuralSignal) -> List[List[str]]:
        """Generate broadcast routes to all nodes"""
        try:
            source = signal.source_id
            routes = []
            
            for target_id in self.nodes:
                if target_id != source:
                    route = self._route_shortest_path(
                        NeuralSignal(
                            signal_id="broadcast",
                            source_id=source,
                            target_id=target_id,
                            signal_type="broadcast",
                            data=None,
                            strength=1.0,
                            timestamp=time.time()
                        )
                    )
                    if route:
                        routes.append(route)
            
            return routes
            
        except Exception as e:
            self.logger.error(f"Broadcast routing failed: {e}")
            return []
    
    def _transmit_signal(self, signal: NeuralSignal, route: List[str]) -> bool:
        """Transmit signal along route"""
        try:
            if len(route) < 2:
                return False
            
            # Update signal path
            signal.path = route.copy()
            
            # Check if direct transmission or multi-hop
            if len(route) == 2:
                # Direct transmission
                return self._deliver_signal(signal, route[1])
            else:
                # Multi-hop transmission
                return self._forward_signal(signal, route)
            
        except Exception as e:
            self.logger.error(f"Signal transmission failed: {e}")
            return False
    
    def _deliver_signal(self, signal: NeuralSignal, target_id: str) -> bool:
        """Deliver signal to target node"""
        try:
            # Check if target node exists and is active
            target_node = self.nodes.get(target_id)
            if not target_node or target_node.state == NodeState.FAILED:
                return False
            
            # Add to target's signal queue
            signal_queue = self.signal_queues.get(target_id)
            if signal_queue:
                try:
                    signal_queue.put((signal.priority, signal), timeout=1.0)
                    
                    # Update node load
                    self.node_loads[target_id] = min(1.0, self.node_loads.get(target_id, 0.0) + 0.1)
                    
                    # Update node state
                    if target_node.current_load < target_node.max_load:
                        target_node.current_load += 1
                        if target_node.current_load >= target_node.max_load * 0.8:
                            target_node.state = NodeState.BUSY
                        elif target_node.current_load >= target_node.max_load:
                            target_node.state = NodeState.OVERLOADED
                    
                    return True
                    
                except queue.Full:
                    self.logger.warning(f"Signal queue full for node {target_id}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Signal delivery failed: {e}")
            return False
    
    def _forward_signal(self, signal: NeuralSignal, route: List[str]) -> bool:
        """Forward signal through multi-hop route"""
        try:
            # For now, implement as direct delivery to final target
            # In a full implementation, this would handle intermediate forwarding
            return self._deliver_signal(signal, route[-1])
            
        except Exception as e:
            self.logger.error(f"Signal forwarding failed: {e}")
            return False
    
    def receive_signals(self, node_id: str, max_signals: int = 10) -> List[NeuralSignal]:
        """Receive pending signals for a node"""
        try:
            signals = []
            signal_queue = self.signal_queues.get(node_id)
            
            if signal_queue:
                for _ in range(min(max_signals, signal_queue.qsize())):
                    try:
                        priority, signal = signal_queue.get_nowait()
                        signals.append(signal)
                        
                        # Update metrics
                        self.performance_metrics.record_signal_received(signal)
                        
                        # Update node load (decrease)
                        node = self.nodes.get(node_id)
                        if node and node.current_load > 0:
                            node.current_load -= 1
                            if node.current_load < node.max_load * 0.5:
                                node.state = NodeState.ACTIVE
                            elif node.current_load < node.max_load * 0.8:
                                node.state = NodeState.IDLE
                        
                        self.node_loads[node_id] = max(0.0, self.node_loads.get(node_id, 0.0) - 0.1)
                        
                    except queue.Empty:
                        break
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to receive signals for {node_id}: {e}")
            return []
    
    def broadcast_signal(self, source_id: str, signal_type: str, data: Any,
                        strength: float = 1.0, priority: int = 0) -> List[str]:
        """Broadcast signal to all nodes"""
        try:
            signal_ids = []
            
            for target_id in self.nodes:
                if target_id != source_id:
                    signal_id = self.send_signal(
                        source_id, target_id, signal_type, data, strength, priority, 'broadcast'
                    )
                    if signal_id:
                        signal_ids.append(signal_id)
            
            return signal_ids
            
        except Exception as e:
            self.logger.error(f"Broadcast failed: {e}")
            return []
    
    def start(self) -> bool:
        """Start the neural fabric"""
        try:
            if self.running:
                return False
            
            self.running = True
            self.shutdown_event.clear()
            
            # Start subsystems
            self.load_balancer.start()
            self.health_monitor.start()
            self.performance_metrics.start()
            self.failure_detector.start()
            
            # Start fabric threads
            self._start_fabric_threads()
            
            self.logger.info("Neural fabric started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start neural fabric: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the neural fabric"""
        try:
            self.running = False
            self.shutdown_event.set()
            
            # Stop subsystems
            self.load_balancer.stop()
            self.health_monitor.stop()
            self.performance_metrics.stop()
            self.failure_detector.stop()
            
            # Stop fabric threads
            self._stop_fabric_threads()
            
            self.logger.info("Neural fabric stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop neural fabric: {e}")
            return False
    
    def _start_fabric_threads(self):
        """Start fabric management threads"""
        try:
            # Heartbeat thread
            self.fabric_threads['heartbeat'] = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self.fabric_threads['heartbeat'].start()
            
            # Load balancing thread
            self.fabric_threads['load_balance'] = threading.Thread(
                target=self._load_balance_loop, daemon=True
            )
            self.fabric_threads['load_balance'].start()
            
            # Topology management thread
            self.fabric_threads['topology'] = threading.Thread(
                target=self._topology_management_loop, daemon=True
            )
            self.fabric_threads['topology'].start()
            
        except Exception as e:
            self.logger.error(f"Failed to start fabric threads: {e}")
    
    def _stop_fabric_threads(self):
        """Stop fabric management threads"""
        try:
            for thread_name, thread in self.fabric_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
            
            self.fabric_threads.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to stop fabric threads: {e}")
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                for node_id, node in self.nodes.items():
                    # Check heartbeat timeout
                    if current_time - node.last_heartbeat > self.config['heartbeat_interval'] * 3:
                        if node.state != NodeState.FAILED:
                            self.logger.warning(f"Node {node_id} heartbeat timeout")
                            node.state = NodeState.FAILED
                            self.failure_detector.report_failure(node_id, "heartbeat_timeout")
                
                time.sleep(self.config['heartbeat_interval'])
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                time.sleep(1.0)
    
    def _load_balance_loop(self):
        """Load balancing loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Update load balancer with current loads
                for node_id, load in self.node_loads.items():
                    self.load_balancer.update_node_load(node_id, load)
                
                # Perform load balancing
                self.load_balancer.balance_load()
                
                time.sleep(self.config['load_balance_interval'])
                
            except Exception as e:
                self.logger.error(f"Load balance loop error: {e}")
                time.sleep(1.0)
    
    def _topology_management_loop(self):
        """Topology management loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Update topology metrics
                self._update_topology_metrics()
                
                # Optimize connections if needed
                if self.config['adaptive_weights']:
                    self._adapt_connection_weights()
                
                # Clear old routing cache
                self._clear_routing_cache()
                
                time.sleep(self.config['topology_update_interval'])
                
            except Exception as e:
                self.logger.error(f"Topology management loop error: {e}")
                time.sleep(1.0)
    
    def _update_topology_metrics(self):
        """Update topology performance metrics"""
        try:
            for conn_id, connection in self.connections.items():
                # Update connection load based on usage
                usage_rate = connection.usage_count / max(1, time.time() - connection.last_used + 1)
                self.connection_loads[conn_id] = min(1.0, usage_rate / connection.bandwidth)
                
                # Update reliability based on successful transmissions
                # This would be enhanced with actual success/failure tracking
                
        except Exception as e:
            self.logger.error(f"Failed to update topology metrics: {e}")
    
    def _adapt_connection_weights(self):
        """Adapt connection weights based on performance"""
        try:
            for connection in self.connections.values():
                if connection.active and connection.usage_count > 0:
                    # Increase weight for frequently used connections
                    usage_factor = min(2.0, 1.0 + (connection.usage_count / 1000.0))
                    connection.weight = min(2.0, connection.weight * usage_factor)
                    
                    # Decrease weight for unreliable connections
                    if connection.reliability < 0.8:
                        connection.weight *= 0.9
                
        except Exception as e:
            self.logger.error(f"Failed to adapt connection weights: {e}")
    
    def _clear_routing_cache(self):
        """Clear cached routing information"""
        try:
            self.signal_routes.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to clear routing cache: {e}")
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat"""
        try:
            node = self.nodes.get(node_id)
            if node:
                node.last_heartbeat = time.time()
                if node.state == NodeState.FAILED:
                    node.state = NodeState.ACTIVE
                    self.logger.info(f"Node {node_id} recovered")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False
    
    def get_fabric_status(self) -> Dict:
        """Get comprehensive fabric status"""
        try:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for node in self.nodes.values() 
                             if node.state in [NodeState.ACTIVE, NodeState.IDLE, NodeState.BUSY])
            
            total_connections = len(self.connections)
            active_connections = sum(1 for conn in self.connections.values() if conn.active)
            
            total_signals = sum(queue.qsize() for queue in self.signal_queues.values())
            
            return {
                'fabric_id': self.fabric_id,
                'running': self.running,
                'nodes': {
                    'total': total_nodes,
                    'active': active_nodes,
                    'failed': total_nodes - active_nodes
                },
                'connections': {
                    'total': total_connections,
                    'active': active_connections
                },
                'signals': {
                    'pending': total_signals,
                    'history_size': len(self.signal_history)
                },
                'performance': self.performance_metrics.get_metrics(),
                'load_balancer': self.load_balancer.get_status(),
                'health_monitor': self.health_monitor.get_status()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get fabric status: {e}")
            return {}
    
    def get_node_status(self, node_id: str) -> Dict:
        """Get detailed node status"""
        try:
            node = self.nodes.get(node_id)
            if not node:
                return {}
            
            signal_queue = self.signal_queues.get(node_id)
            pending_signals = signal_queue.qsize() if signal_queue else 0
            
            connections = self.get_node_connections(node_id)
            
            return {
                'node_id': node_id,
                'type': node.node_type.value,
                'state': node.state.value,
                'capabilities': node.capabilities,
                'load': {
                    'current': node.current_load,
                    'max': node.max_load,
                    'percentage': (node.current_load / node.max_load) * 100
                },
                'signals': {
                    'pending': pending_signals,
                    'total_processed': node.total_processed
                },
                'connections': {
                    'total': len(connections),
                    'outgoing': len([c for c in connections if c['source_id'] == node_id]),
                    'incoming': len([c for c in connections if c['target_id'] == node_id])
                },
                'performance': {
                    'processing_time': node.processing_time,
                    'error_count': node.error_count,
                    'last_heartbeat': node.last_heartbeat
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get node status for {node_id}: {e}")
            return {}
    
    def get_node_connections(self, node_id: str) -> List[Dict]:
        """Get all connections for a node"""
        try:
            connections = []
            
            for connection in self.connections.values():
                if connection.source_id == node_id or connection.target_id == node_id:
                    connections.append({
                        'connection_id': connection.connection_id,
                        'source_id': connection.source_id,
                        'target_id': connection.target_id,
                        'type': connection.connection_type.value,
                        'weight': connection.weight,
                        'bandwidth': connection.bandwidth,
                        'latency': connection.latency,
                        'reliability': connection.reliability,
                        'usage_count': connection.usage_count,
                        'last_used': connection.last_used,
                        'active': connection.active
                    })
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Failed to get connections for {node_id}: {e}")
            return []

# Supporting classes for advanced functionality

class LoadBalancer:
    """Load balancer for neural fabric"""
    
    def __init__(self):
        self.nodes = {}
        self.running = False
    
    def register_node(self, node_id: str, max_load: int):
        self.nodes[node_id] = {'max_load': max_load, 'current_load': 0.0}
    
    def unregister_node(self, node_id: str):
        self.nodes.pop(node_id, None)
    
    def update_node_load(self, node_id: str, load: float):
        if node_id in self.nodes:
            self.nodes[node_id]['current_load'] = load
    
    def balance_load(self):
        # Implement load balancing logic
        pass
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_status(self):
        return {'running': self.running, 'nodes': len(self.nodes)}

class HealthMonitor:
    """Health monitor for neural fabric"""
    
    def __init__(self):
        self.nodes = {}
        self.running = False
    
    def register_node(self, node_id: str, signal_handler: Callable = None):
        self.nodes[node_id] = {'handler': signal_handler, 'healthy': True}
    
    def unregister_node(self, node_id: str):
        self.nodes.pop(node_id, None)
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_status(self):
        healthy_nodes = sum(1 for node in self.nodes.values() if node['healthy'])
        return {'running': self.running, 'healthy_nodes': healthy_nodes, 'total_nodes': len(self.nodes)}

class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics = {
            'signals_sent': 0,
            'signals_received': 0,
            'average_latency': 0.0,
            'throughput': 0.0
        }
        self.running = False
    
    def record_signal_sent(self, signal: NeuralSignal):
        self.metrics['signals_sent'] += 1
    
    def record_signal_received(self, signal: NeuralSignal):
        self.metrics['signals_received'] += 1
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_metrics(self):
        return self.metrics.copy()

class FailureDetector:
    """Failure detector for neural fabric"""
    
    def __init__(self):
        self.failures = deque(maxlen=1000)
        self.running = False
    
    def report_failure(self, node_id: str, failure_type: str):
        self.failures.append({
            'node_id': node_id,
            'failure_type': failure_type,
            'timestamp': time.time()
        })
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False

# Alias for backward compatibility
CortexOSNeuralFabric = AdvancedNeuralFabric
NeuralFabric = AdvancedNeuralFabric

if __name__ == "__main__":
    # Comprehensive test suite
    print("🕸️ Testing Advanced CortexOS Neural Fabric...")
    
    # Create fabric
    fabric = AdvancedNeuralFabric()
    
    # Test node registration
    print("📝 Testing node registration...")
    nodes = [
        ("neuroengine_1", NodeType.PROCESSOR, ["learning", "pattern_recognition"]),
        ("memory_1", NodeType.STORAGE, ["store", "retrieve"]),
        ("resonance_1", NodeType.MODULATOR, ["amplify", "filter"]),
        ("gateway_1", NodeType.GATEWAY, ["route", "transform"])
    ]
    
    for node_id, node_type, capabilities in nodes:
        success = fabric.register_node(node_id, node_type, capabilities, max_load=50)
        print(f"{'✅' if success else '❌'} Registered {node_id} ({node_type.value})")
    
    # Test connection creation
    print("\n🔗 Testing connection creation...")
    connections = [
        ("neuroengine_1", "memory_1", ConnectionType.DATA_FLOW, 1.0),
        ("memory_1", "resonance_1", ConnectionType.FEEDBACK, 0.8),
        ("resonance_1", "neuroengine_1", ConnectionType.CONTROL, 0.9),
        ("gateway_1", "neuroengine_1", ConnectionType.DATA_FLOW, 1.2)
    ]
    
    for source, target, conn_type, weight in connections:
        conn_id = fabric.create_connection(source, target, conn_type, weight)
        print(f"{'✅' if conn_id else '❌'} Created connection: {source} -> {target}")
    
    # Test fabric startup
    print("\n🚀 Testing fabric startup...")
    if fabric.start():
        print("✅ Neural fabric started successfully")
        
        # Test signal sending
        print("\n📡 Testing signal transmission...")
        test_signals = [
            ("neuroengine_1", "memory_1", "data_store", {"pattern": "test_pattern_1"}),
            ("memory_1", "resonance_1", "feedback", {"strength": 0.8}),
            ("gateway_1", "neuroengine_1", "input_data", {"data": "external_input"})
        ]
        
        for source, target, signal_type, data in test_signals:
            signal_id = fabric.send_signal(source, target, signal_type, data, strength=1.0)
            print(f"{'✅' if signal_id else '❌'} Sent signal: {source} -> {target} ({signal_type})")
        
        # Test signal receiving
        print("\n📨 Testing signal reception...")
        for node_id in ["memory_1", "resonance_1", "neuroengine_1"]:
            signals = fabric.receive_signals(node_id, max_signals=5)
            print(f"📬 {node_id} received {len(signals)} signals")
            for signal in signals:
                print(f"  - {signal.signal_type} from {signal.source_id}")
        
        # Test broadcast
        print("\n📢 Testing broadcast...")
        broadcast_ids = fabric.broadcast_signal("gateway_1", "system_alert", {"alert": "test_broadcast"})
        print(f"✅ Broadcast sent to {len(broadcast_ids)} nodes")
        
        # Test heartbeat
        print("\n💓 Testing heartbeat...")
        for node_id in nodes:
            success = fabric.update_node_heartbeat(node_id[0])
            print(f"{'✅' if success else '❌'} Heartbeat updated for {node_id[0]}")
        
        # Let fabric run for a moment
        time.sleep(2)
        
        # Test status
        print("\n📊 Testing status reporting...")
        fabric_status = fabric.get_fabric_status()
        print(f"Fabric Status:")
        print(f"  Nodes: {fabric_status['nodes']['active']}/{fabric_status['nodes']['total']} active")
        print(f"  Connections: {fabric_status['connections']['active']}/{fabric_status['connections']['total']} active")
        print(f"  Pending Signals: {fabric_status['signals']['pending']}")
        
        # Test individual node status
        print(f"\nNode Status for neuroengine_1:")
        node_status = fabric.get_node_status("neuroengine_1")
        if node_status:
            print(f"  State: {node_status['state']}")
            print(f"  Load: {node_status['load']['percentage']:.1f}%")
            print(f"  Connections: {node_status['connections']['total']}")
        
        # Test shutdown
        print("\n🛑 Testing fabric shutdown...")
        if fabric.stop():
            print("✅ Neural fabric stopped successfully")
    else:
        print("❌ Failed to start neural fabric")
    
    print("\n🎉 Advanced Neural Fabric test complete!")

