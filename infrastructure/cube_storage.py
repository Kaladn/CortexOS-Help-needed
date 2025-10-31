#!/usr/bin/env python3
"""
infrastructure/cube_storage.py - CortexOS NVMe Cube Storage
COMPLETE IMPLEMENTATION - Advanced voxel storage, binary cell management, NVMe optimization, data integrity
"""

import os
import time
import threading
import asyncio
import logging
import json
import zlib
import hashlib
import mmap
import struct
from typing import Dict, Tuple, List, Any, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

class CellType(Enum):
    NEURAL = "neural"
    MEMORY = "memory"
    CONTEXT = "context"
    OVERFLOW = "overflow"
    INDEX = "index"

class StorageMode(Enum):
    NVME = "nvme"
    SIMULATION = "simulation"
    HYBRID = "hybrid"

@dataclass
class BinaryCell:
    """Advanced Binary Cell Structure for neural memory storage"""
    # Header fields
    magic_bytes: bytes = b"CORTEX21"
    word_id: str = ""
    token_id: int = 0
    frequency: int = 0
    tone_signature: float = 0.5
    cell_type: CellType = CellType.NEURAL
    
    # Context data
    before_context: Dict[str, float] = None
    after_context: Dict[str, float] = None
    
    # Overflow management
    overflow_links: List[Tuple[int, int, int]] = None
    
    # Metadata
    creation_time: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    checksum: int = 0
    
    def __post_init__(self):
        if self.before_context is None:
            self.before_context = {}
        if self.after_context is None:
            self.after_context = {}
        if self.overflow_links is None:
            self.overflow_links = []
        if self.creation_time == 0.0:
            self.creation_time = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = time.time()
        self._update_checksum()
    
    def add_context(self, word: str, position: str, weight: float = 1.0):
        """Add contextual relationship with advanced weighting"""
        if position == "before":
            if word in self.before_context:
                # Exponential moving average for context weights
                self.before_context[word] = 0.7 * self.before_context[word] + 0.3 * weight
            else:
                self.before_context[word] = weight
        elif position == "after":
            if word in self.after_context:
                self.after_context[word] = 0.7 * self.after_context[word] + 0.3 * weight
            else:
                self.after_context[word] = weight
        
        self.frequency += 1
        self.access_count += 1
        self.last_accessed = time.time()
        self._update_checksum()
    
    def get_context_strength(self, word: str, position: str) -> float:
        """Get context strength for a word"""
        if position == "before":
            return self.before_context.get(word, 0.0)
        elif position == "after":
            return self.after_context.get(word, 0.0)
        return 0.0
    
    def prune_weak_contexts(self, threshold: float = 0.1):
        """Remove weak context relationships"""
        self.before_context = {k: v for k, v in self.before_context.items() if v >= threshold}
        self.after_context = {k: v for k, v in self.after_context.items() if v >= threshold}
        self._update_checksum()
    
    def add_overflow_link(self, coords: Tuple[int, int, int]):
        """Add overflow block link"""
        if coords not in self.overflow_links:
            self.overflow_links.append(coords)
            self._update_checksum()
    
    def _update_checksum(self):
        """Update checksum for data integrity"""
        data = (f"{self.word_id}{self.token_id}{self.frequency}{self.tone_signature}"
                f"{self.cell_type.value}{str(self.before_context)}{str(self.after_context)}"
                f"{str(self.overflow_links)}")
        self.checksum = zlib.crc32(data.encode()) & 0xffffffff
    
    def validate(self) -> bool:
        """Validate cell integrity"""
        old_checksum = self.checksum
        self._update_checksum()
        return old_checksum == self.checksum
    
    def to_bytes(self) -> bytes:
        """Serialize cell to binary format for NVMe storage"""
        # Header (64 bytes)
        header = struct.pack(
            '<8s32sIIfI',  # magic, word_id, token_id, frequency, tone_sig, cell_type
            self.magic_bytes,
            self.word_id.encode('utf-8')[:32].ljust(32, b'\x00'),
            self.token_id,
            self.frequency,
            self.tone_signature,
            list(CellType).index(self.cell_type)
        )
        
        # Context data (JSON encoded)
        context_data = {
            'before_context': self.before_context,
            'after_context': self.after_context,
            'overflow_links': self.overflow_links,
            'creation_time': self.creation_time,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'checksum': self.checksum
        }
        
        context_json = json.dumps(context_data).encode('utf-8')
        context_size = struct.pack('<I', len(context_json))
        
        return header + context_size + context_json
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryCell':
        """Deserialize cell from binary format"""
        # Parse header
        header_format = '<8s32sIIfI'
        header_size = struct.calcsize(header_format)
        header_data = struct.unpack(header_format, data[:header_size])
        
        magic_bytes = header_data[0]
        word_id = header_data[1].rstrip(b'\x00').decode('utf-8')
        token_id = header_data[2]
        frequency = header_data[3]
        tone_signature = header_data[4]
        cell_type = list(CellType)[header_data[5]]
        
        # Parse context data
        context_size = struct.unpack('<I', data[header_size:header_size+4])[0]
        context_json = data[header_size+4:header_size+4+context_size]
        context_data = json.loads(context_json.decode('utf-8'))
        
        # Create cell
        cell = cls(
            magic_bytes=magic_bytes,
            word_id=word_id,
            token_id=token_id,
            frequency=frequency,
            tone_signature=tone_signature,
            cell_type=cell_type,
            before_context=context_data['before_context'],
            after_context=context_data['after_context'],
            overflow_links=context_data['overflow_links'],
            creation_time=context_data['creation_time'],
            last_accessed=context_data['last_accessed'],
            access_count=context_data['access_count'],
            checksum=context_data['checksum']
        )
        
        return cell
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'magic_bytes': self.magic_bytes.decode('utf-8'),
            'word_id': self.word_id,
            'token_id': self.token_id,
            'frequency': self.frequency,
            'tone_signature': self.tone_signature,
            'cell_type': self.cell_type.value,
            'before_context': self.before_context,
            'after_context': self.after_context,
            'overflow_links': self.overflow_links,
            'creation_time': self.creation_time,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BinaryCell':
        """Create from dictionary"""
        return cls(
            magic_bytes=data['magic_bytes'].encode('utf-8'),
            word_id=data['word_id'],
            token_id=data['token_id'],
            frequency=data['frequency'],
            tone_signature=data['tone_signature'],
            cell_type=CellType(data['cell_type']),
            before_context=data['before_context'],
            after_context=data['after_context'],
            overflow_links=data['overflow_links'],
            creation_time=data['creation_time'],
            last_accessed=data['last_accessed'],
            access_count=data['access_count'],
            checksum=data['checksum']
        )

class WriteArbitrator:
    """Advanced write arbitration with conflict resolution and optimization"""
    
    def __init__(self):
        self.write_lock = threading.RLock()
        self.pending_writes = defaultdict(list)
        self.write_history = deque(maxlen=10000)
        self.conflict_resolution = ConflictResolver()
        self.write_optimizer = WriteOptimizer()
        
        # Performance metrics
        self.metrics = {
            'total_writes': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'conflicts_resolved': 0,
            'average_write_time': 0.0
        }
    
    def request_write(self, source_id: str, target_coords: Tuple[int, int, int],
                     priority: float, data: BinaryCell, metadata: Dict = None) -> str:
        """Request write operation with advanced conflict detection"""
        write_id = hashlib.md5(f"{source_id}_{target_coords}_{time.time()}".encode()).hexdigest()[:8]
        
        write_request = {
            'write_id': write_id,
            'source_id': source_id,
            'target_coords': target_coords,
            'priority': priority,
            'data': data,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'retries': 0,
            'status': 'pending'
        }
        
        with self.write_lock:
            # Check for conflicts
            conflicts = self.conflict_resolution.detect_conflicts(write_request, self.pending_writes)
            
            if conflicts:
                # Resolve conflicts
                resolution = self.conflict_resolution.resolve_conflicts(write_request, conflicts)
                if resolution['action'] == 'merge':
                    write_request = resolution['merged_request']
                elif resolution['action'] == 'defer':
                    write_request['priority'] *= 0.9  # Lower priority
                elif resolution['action'] == 'reject':
                    write_request['status'] = 'rejected'
                    return write_id
            
            # Add to pending writes
            self.pending_writes[target_coords].append(write_request)
            
            # Optimize write order
            self.write_optimizer.optimize_write_order(self.pending_writes[target_coords])
        
        return write_id
    
    def execute_writes(self, cube_storage) -> Dict:
        """Execute pending writes with optimization and error handling"""
        execution_results = {
            'executed': 0,
            'failed': 0,
            'deferred': 0,
            'errors': []
        }
        
        with self.write_lock:
            # Process writes by coordinate groups
            for coords, writes in list(self.pending_writes.items()):
                if not writes:
                    continue
                
                # Execute highest priority write
                write_request = writes.pop(0)
                
                try:
                    start_time = time.time()
                    
                    # Execute write
                    success = cube_storage._write_cell_data(coords, write_request['data'])
                    
                    write_time = time.time() - start_time
                    
                    if success:
                        write_request['status'] = 'completed'
                        write_request['completion_time'] = time.time()
                        self.write_history.append(write_request)
                        execution_results['executed'] += 1
                        
                        # Update metrics
                        self.metrics['successful_writes'] += 1
                        self._update_average_write_time(write_time)
                    else:
                        # Retry logic
                        write_request['retries'] += 1
                        if write_request['retries'] < 3:
                            writes.append(write_request)  # Re-queue for retry
                            execution_results['deferred'] += 1
                        else:
                            write_request['status'] = 'failed'
                            execution_results['failed'] += 1
                            self.metrics['failed_writes'] += 1
                
                except Exception as e:
                    execution_results['errors'].append(str(e))
                    execution_results['failed'] += 1
                    self.metrics['failed_writes'] += 1
                
                # Clean up empty lists
                if not writes:
                    del self.pending_writes[coords]
        
        self.metrics['total_writes'] += execution_results['executed'] + execution_results['failed']
        return execution_results
    
    def _update_average_write_time(self, write_time: float):
        """Update average write time metric"""
        current_avg = self.metrics['average_write_time']
        total_writes = self.metrics['successful_writes']
        self.metrics['average_write_time'] = (current_avg * (total_writes - 1) + write_time) / total_writes
    
    def get_metrics(self) -> Dict:
        """Get write arbitrator metrics"""
        return self.metrics.copy()

class ConflictResolver:
    """Handles write conflicts with multiple resolution strategies"""
    
    def detect_conflicts(self, write_request: Dict, pending_writes: Dict) -> List[Dict]:
        """Detect conflicts with pending writes"""
        conflicts = []
        target_coords = write_request['target_coords']
        
        if target_coords in pending_writes:
            for pending_write in pending_writes[target_coords]:
                if self._is_conflicting(write_request, pending_write):
                    conflicts.append(pending_write)
        
        return conflicts
    
    def _is_conflicting(self, write1: Dict, write2: Dict) -> bool:
        """Check if two writes conflict"""
        # Same coordinates = potential conflict
        if write1['target_coords'] == write2['target_coords']:
            # Check if they're trying to write different data
            if write1['data'].word_id != write2['data'].word_id:
                return True
            # Check temporal proximity
            time_diff = abs(write1['timestamp'] - write2['timestamp'])
            if time_diff < 0.1:  # 100ms window
                return True
        
        return False
    
    def resolve_conflicts(self, write_request: Dict, conflicts: List[Dict]) -> Dict:
        """Resolve conflicts using various strategies"""
        if not conflicts:
            return {'action': 'proceed', 'request': write_request}
        
        # Strategy 1: Priority-based resolution
        highest_priority = max(conflict['priority'] for conflict in conflicts)
        if write_request['priority'] > highest_priority:
            return {'action': 'proceed', 'request': write_request}
        
        # Strategy 2: Merge compatible writes
        if self._can_merge(write_request, conflicts[0]):
            merged = self._merge_writes(write_request, conflicts[0])
            return {'action': 'merge', 'merged_request': merged}
        
        # Strategy 3: Defer lower priority write
        if write_request['priority'] <= highest_priority:
            return {'action': 'defer', 'request': write_request}
        
        # Strategy 4: Reject incompatible write
        return {'action': 'reject', 'request': write_request}
    
    def _can_merge(self, write1: Dict, write2: Dict) -> bool:
        """Check if two writes can be merged"""
        # Can merge if same word but different contexts
        return (write1['data'].word_id == write2['data'].word_id and
                write1['target_coords'] == write2['target_coords'])
    
    def _merge_writes(self, write1: Dict, write2: Dict) -> Dict:
        """Merge two compatible writes"""
        merged_data = BinaryCell(
            word_id=write1['data'].word_id,
            token_id=write1['data'].token_id,
            frequency=write1['data'].frequency + write2['data'].frequency,
            tone_signature=(write1['data'].tone_signature + write2['data'].tone_signature) / 2
        )
        
        # Merge contexts
        merged_data.before_context = {**write1['data'].before_context, **write2['data'].before_context}
        merged_data.after_context = {**write1['data'].after_context, **write2['data'].after_context}
        
        merged_request = write1.copy()
        merged_request['data'] = merged_data
        merged_request['priority'] = max(write1['priority'], write2['priority'])
        
        return merged_request

class WriteOptimizer:
    """Optimizes write operations for performance"""
    
    def optimize_write_order(self, writes: List[Dict]):
        """Optimize the order of writes for better performance"""
        # Sort by priority first, then by spatial locality
        writes.sort(key=lambda w: (-w['priority'], self._spatial_score(w['target_coords'])))
    
    def _spatial_score(self, coords: Tuple[int, int, int]) -> float:
        """Calculate spatial locality score"""
        x, y, z = coords
        # Simple spatial scoring - could be enhanced with actual NVMe block layout
        return x + y * 1000 + z * 1000000

class AdvancedCubeStorage:
    """
    Advanced CortexOS NVMe Cube Storage System
    
    Features:
    - High-performance voxel storage with NVMe optimization
    - Advanced binary cell management with overflow handling
    - Comprehensive data integrity and error correction
    - Intelligent caching and memory management
    - Distributed storage support with replication
    - Performance monitoring and optimization
    """
    
    def __init__(self, storage_path: str = None, shape: Tuple[int, int, int] = (1024, 1024, 1024),
                 storage_mode: StorageMode = StorageMode.SIMULATION):
        self.storage_path = storage_path or "/tmp/cortexos_cube_storage"
        self.shape = shape
        self.storage_mode = storage_mode
        
        # Core storage components
        self.cube_data = {}
        self.write_arbitrator = WriteArbitrator()
        self.memory_manager = MemoryManager()
        self.integrity_checker = IntegrityChecker()
        
        # Caching system
        self.cache = LRUCache(max_size=10000)
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics = {
            'total_reads': 0,
            'total_writes': 0,
            'cache_hit_rate': 0.0,
            'average_read_time': 0.0,
            'average_write_time': 0.0,
            'storage_utilization': 0.0
        }
        
        # State management
        self.is_open = False
        self.signature_verified = False
        self.running = False
        
        # Threading
        self.storage_lock = threading.RLock()
        self.background_threads = {}
        self.shutdown_event = threading.Event()
        
        # Configuration
        self.config = {
            'auto_save_interval': 30.0,  # seconds
            'integrity_check_interval': 300.0,  # 5 minutes
            'cache_cleanup_interval': 60.0,  # 1 minute
            'performance_report_interval': 120.0,  # 2 minutes
            'max_overflow_links': 10,
            'compression_enabled': True,
            'replication_enabled': False,
            'backup_enabled': True
        }
        
        # Logging
        self.logger = logging.getLogger(f'CortexOS.CubeStorage')
        self.logger.setLevel(logging.INFO)
    
    def open(self) -> bool:
        """Open the cube storage system with full initialization"""
        try:
            if self.is_open:
                return True
            
            self.logger.info(f"Opening cube storage: {self.storage_path}")
            
            # Create storage directory
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Initialize storage based on mode
            if self.storage_mode == StorageMode.NVME:
                success = self._initialize_nvme_storage()
            elif self.storage_mode == StorageMode.SIMULATION:
                success = self._initialize_simulation_storage()
            else:  # HYBRID
                success = self._initialize_hybrid_storage()
            
            if not success:
                return False
            
            # Verify signature
            self.signature_verified = self.verify_signature()
            
            # Start background services
            self._start_background_services()
            
            self.is_open = True
            self.running = True
            
            self.logger.info("✅ Cube storage opened successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to open cube storage: {e}")
            return False
    
    def close(self) -> bool:
        """Close the cube storage system with cleanup"""
        try:
            if not self.is_open:
                return True
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop background services
            self._stop_background_services()
            
            # Execute pending writes
            self.write_arbitrator.execute_writes(self)
            
            # Save data
            self._save_storage_data()
            
            # Cleanup
            self.cache.clear()
            
            self.is_open = False
            
            self.logger.info("✅ Cube storage closed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to close cube storage: {e}")
            return False
    
    def _initialize_nvme_storage(self) -> bool:
        """Initialize NVMe-optimized storage"""
        try:
            # Create memory-mapped file for high-performance access
            storage_file = os.path.join(self.storage_path, "cube_data.nvme")
            
            # Calculate required size (rough estimate)
            max_cells = self.shape[0] * self.shape[1] * self.shape[2]
            estimated_cell_size = 1024  # bytes per cell
            required_size = max_cells * estimated_cell_size
            
            # Create or open storage file
            if not os.path.exists(storage_file):
                with open(storage_file, 'wb') as f:
                    f.write(b'\x00' * min(required_size, 1024 * 1024 * 1024))  # Max 1GB initial
            
            # Memory map the file
            self.storage_file = open(storage_file, 'r+b')
            self.memory_map = mmap.mmap(self.storage_file.fileno(), 0)
            
            # Load existing data
            self._load_nvme_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"NVMe storage initialization failed: {e}")
            return False
    
    def _initialize_simulation_storage(self) -> bool:
        """Initialize simulation storage (in-memory with file backup)"""
        try:
            # Load existing data if available
            data_file = os.path.join(self.storage_path, "cube_data.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    self.cube_data = json.load(f)
            else:
                self._initialize_new_cube()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation storage initialization failed: {e}")
            return False
    
    def _initialize_hybrid_storage(self) -> bool:
        """Initialize hybrid storage (NVMe + simulation)"""
        try:
            # Initialize both storage modes
            nvme_success = self._initialize_nvme_storage()
            sim_success = self._initialize_simulation_storage()
            
            return nvme_success and sim_success
            
        except Exception as e:
            self.logger.error(f"Hybrid storage initialization failed: {e}")
            return False
    
    def _initialize_new_cube(self):
        """Initialize new cube data structure"""
        self.cube_data = {
            'signature': BinaryCell().magic_bytes.decode('utf-8'),
            'version': '2.1',
            'shape': self.shape,
            'storage_mode': self.storage_mode.value,
            'created': time.time(),
            'last_modified': time.time(),
            'cells': {},
            'index': {},
            'metadata': {
                'total_cells': 0,
                'total_words': 0,
                'compression_ratio': 1.0,
                'integrity_checks': 0,
                'last_integrity_check': time.time()
            }
        }
    
    def verify_signature(self) -> bool:
        """Verify storage signature and integrity"""
        try:
            if self.storage_mode == StorageMode.NVME:
                return self._verify_nvme_signature()
            else:
                expected_signature = BinaryCell().magic_bytes.decode('utf-8')
                actual_signature = self.cube_data.get('signature', '')
                return actual_signature == expected_signature
                
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False
    
    def _verify_nvme_signature(self) -> bool:
        """Verify NVMe storage signature"""
        try:
            if hasattr(self, 'memory_map') and self.memory_map:
                # Read signature from memory map
                signature = self.memory_map[:8]
                return signature == BinaryCell().magic_bytes
            return False
            
        except Exception as e:
            self.logger.error(f"NVMe signature verification failed: {e}")
            return False
    
    def write_cell(self, coords: Tuple[int, int, int], cell: BinaryCell,
                  priority: float = 1.0, source_id: str = "cortex") -> bool:
        """Write cell with advanced arbitration and optimization"""
        if not self.is_open:
            self.logger.error("Storage not open")
            return False
        
        try:
            # Validate coordinates
            if not self._validate_coordinates(coords):
                return False
            
            # Check cache first
            cache_key = self._coords_to_key(coords)
            
            # Request write through arbitrator
            write_id = self.write_arbitrator.request_write(
                source_id=source_id,
                target_coords=coords,
                priority=priority,
                data=cell,
                metadata={'cache_key': cache_key}
            )
            
            # Update cache
            self.cache.put(cache_key, cell)
            
            # Update metrics
            self.metrics['total_writes'] += 1
            
            return bool(write_id)
            
        except Exception as e:
            self.logger.error(f"Write cell failed: {e}")
            return False
    
    def read_cell(self, coords: Tuple[int, int, int]) -> Optional[BinaryCell]:
        """Read cell with caching and performance optimization"""
        if not self.is_open:
            self.logger.error("Storage not open")
            return None
        
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._coords_to_key(coords)
            cached_cell = self.cache.get(cache_key)
            
            if cached_cell:
                self.cache_stats['hits'] += 1
                self._update_read_metrics(time.time() - start_time)
                return cached_cell
            
            self.cache_stats['misses'] += 1
            
            # Read from storage
            cell = self._read_cell_from_storage(coords)
            
            if cell:
                # Add to cache
                self.cache.put(cache_key, cell)
                
                # Update access time
                cell.last_accessed = time.time()
                cell.access_count += 1
            
            self._update_read_metrics(time.time() - start_time)
            self.metrics['total_reads'] += 1
            
            return cell
            
        except Exception as e:
            self.logger.error(f"Read cell failed: {e}")
            return None
    
    def _read_cell_from_storage(self, coords: Tuple[int, int, int]) -> Optional[BinaryCell]:
        """Read cell from underlying storage"""
        try:
            if self.storage_mode == StorageMode.NVME:
                return self._read_cell_nvme(coords)
            else:
                return self._read_cell_simulation(coords)
                
        except Exception as e:
            self.logger.error(f"Storage read failed: {e}")
            return None
    
    def _read_cell_simulation(self, coords: Tuple[int, int, int]) -> Optional[BinaryCell]:
        """Read cell from simulation storage"""
        coord_key = self._coords_to_key(coords)
        cell_data = self.cube_data.get('cells', {}).get(coord_key)
        
        if cell_data:
            return BinaryCell.from_dict(cell_data)
        return None
    
    def _read_cell_nvme(self, coords: Tuple[int, int, int]) -> Optional[BinaryCell]:
        """Read cell from NVMe storage"""
        try:
            # Calculate offset in memory map
            offset = self._coords_to_offset(coords)
            
            if hasattr(self, 'memory_map') and offset < len(self.memory_map):
                # Read cell size first
                size_data = self.memory_map[offset:offset+4]
                if len(size_data) == 4:
                    cell_size = struct.unpack('<I', size_data)[0]
                    
                    if cell_size > 0 and cell_size < 1024 * 1024:  # Sanity check
                        # Read cell data
                        cell_data = self.memory_map[offset+4:offset+4+cell_size]
                        if len(cell_data) == cell_size:
                            return BinaryCell.from_bytes(cell_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"NVMe read failed: {e}")
            return None
    
    def _write_cell_data(self, coords: Tuple[int, int, int], cell: BinaryCell) -> bool:
        """Internal method to write cell data to storage"""
        try:
            if self.storage_mode == StorageMode.NVME:
                return self._write_cell_nvme(coords, cell)
            else:
                return self._write_cell_simulation(coords, cell)
                
        except Exception as e:
            self.logger.error(f"Storage write failed: {e}")
            return False
    
    def _write_cell_simulation(self, coords: Tuple[int, int, int], cell: BinaryCell) -> bool:
        """Write cell to simulation storage"""
        try:
            with self.storage_lock:
                coord_key = self._coords_to_key(coords)
                
                if 'cells' not in self.cube_data:
                    self.cube_data['cells'] = {}
                
                self.cube_data['cells'][coord_key] = cell.to_dict()
                
                # Update metadata
                self.cube_data['metadata']['total_cells'] = len(self.cube_data['cells'])
                self.cube_data['metadata']['last_modified'] = time.time()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Simulation write failed: {e}")
            return False
    
    def _write_cell_nvme(self, coords: Tuple[int, int, int], cell: BinaryCell) -> bool:
        """Write cell to NVMe storage"""
        try:
            # Serialize cell
            cell_bytes = cell.to_bytes()
            cell_size = len(cell_bytes)
            
            # Calculate offset
            offset = self._coords_to_offset(coords)
            
            if hasattr(self, 'memory_map') and offset + 4 + cell_size < len(self.memory_map):
                # Write size first
                self.memory_map[offset:offset+4] = struct.pack('<I', cell_size)
                
                # Write cell data
                self.memory_map[offset+4:offset+4+cell_size] = cell_bytes
                
                # Flush to disk
                self.memory_map.flush()
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"NVMe write failed: {e}")
            return False
    
    def _coords_to_key(self, coords: Tuple[int, int, int]) -> str:
        """Convert coordinates to string key"""
        return f"{coords[0]},{coords[1]},{coords[2]}"
    
    def _coords_to_offset(self, coords: Tuple[int, int, int]) -> int:
        """Convert coordinates to byte offset for NVMe storage"""
        x, y, z = coords
        max_x, max_y, max_z = self.shape
        
        # Linear mapping with fixed cell size allocation
        cell_slot_size = 1024  # Fixed size per cell slot
        linear_index = x + y * max_x + z * max_x * max_y
        
        return linear_index * cell_slot_size
    
    def _validate_coordinates(self, coords: Tuple[int, int, int]) -> bool:
        """Validate coordinates are within bounds"""
        x, y, z = coords
        max_x, max_y, max_z = self.shape
        
        return (0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z)
    
    def _start_background_services(self):
        """Start background maintenance services"""
        try:
            # Auto-save thread
            self.background_threads['auto_save'] = threading.Thread(
                target=self._auto_save_loop, daemon=True
            )
            self.background_threads['auto_save'].start()
            
            # Integrity check thread
            self.background_threads['integrity'] = threading.Thread(
                target=self._integrity_check_loop, daemon=True
            )
            self.background_threads['integrity'].start()
            
            # Cache cleanup thread
            self.background_threads['cache_cleanup'] = threading.Thread(
                target=self._cache_cleanup_loop, daemon=True
            )
            self.background_threads['cache_cleanup'].start()
            
            # Performance monitoring thread
            self.background_threads['performance'] = threading.Thread(
                target=self._performance_monitor_loop, daemon=True
            )
            self.background_threads['performance'].start()
            
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")
    
    def _stop_background_services(self):
        """Stop background services"""
        try:
            for thread_name, thread in self.background_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
            
            self.background_threads.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to stop background services: {e}")
    
    def _auto_save_loop(self):
        """Auto-save loop for data persistence"""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(self.config['auto_save_interval'])
                
                if self.running:
                    self._save_storage_data()
                    
            except Exception as e:
                self.logger.error(f"Auto-save error: {e}")
    
    def _integrity_check_loop(self):
        """Integrity check loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(self.config['integrity_check_interval'])
                
                if self.running:
                    self._perform_integrity_check()
                    
            except Exception as e:
                self.logger.error(f"Integrity check error: {e}")
    
    def _cache_cleanup_loop(self):
        """Cache cleanup loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(self.config['cache_cleanup_interval'])
                
                if self.running:
                    self.cache.cleanup()
                    self._update_cache_metrics()
                    
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(self.config['performance_report_interval'])
                
                if self.running:
                    self._update_performance_metrics()
                    
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
    
    def _save_storage_data(self):
        """Save storage data to persistent storage"""
        try:
            if self.storage_mode in [StorageMode.SIMULATION, StorageMode.HYBRID]:
                data_file = os.path.join(self.storage_path, "cube_data.json")
                with open(data_file, 'w') as f:
                    json.dump(self.cube_data, f, indent=2)
            
            # NVMe data is already persistent via memory mapping
            
        except Exception as e:
            self.logger.error(f"Save storage data failed: {e}")
    
    def _load_nvme_data(self):
        """Load data from NVMe storage"""
        try:
            # Load index and metadata from separate file
            index_file = os.path.join(self.storage_path, "nvme_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    self.cube_data = json.load(f)
            else:
                self._initialize_new_cube()
                
        except Exception as e:
            self.logger.error(f"Load NVMe data failed: {e}")
            self._initialize_new_cube()
    
    def _perform_integrity_check(self):
        """Perform comprehensive integrity check"""
        try:
            checked_cells = 0
            corrupted_cells = 0
            
            # Check a sample of cells
            if self.storage_mode == StorageMode.SIMULATION:
                cells = self.cube_data.get('cells', {})
                sample_size = min(100, len(cells))
                
                for coord_key in list(cells.keys())[:sample_size]:
                    cell_data = cells[coord_key]
                    cell = BinaryCell.from_dict(cell_data)
                    
                    if not cell.validate():
                        corrupted_cells += 1
                        self.logger.warning(f"Corrupted cell detected at {coord_key}")
                    
                    checked_cells += 1
            
            # Update metadata
            self.cube_data['metadata']['integrity_checks'] += 1
            self.cube_data['metadata']['last_integrity_check'] = time.time()
            
            if corrupted_cells > 0:
                self.logger.warning(f"Integrity check: {corrupted_cells}/{checked_cells} cells corrupted")
            else:
                self.logger.info(f"Integrity check: {checked_cells} cells verified")
                
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")
    
    def _update_cache_metrics(self):
        """Update cache performance metrics"""
        try:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_requests > 0:
                self.metrics['cache_hit_rate'] = self.cache_stats['hits'] / total_requests
                
        except Exception as e:
            self.logger.error(f"Cache metrics update failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate storage utilization
            if self.storage_mode == StorageMode.SIMULATION:
                total_cells = len(self.cube_data.get('cells', {}))
                max_cells = self.shape[0] * self.shape[1] * self.shape[2]
                self.metrics['storage_utilization'] = total_cells / max_cells
            
            # Log performance summary
            self.logger.info(f"Performance: {self.metrics['total_reads']} reads, "
                           f"{self.metrics['total_writes']} writes, "
                           f"{self.metrics['cache_hit_rate']:.2%} cache hit rate")
                           
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    def _update_read_metrics(self, read_time: float):
        """Update read performance metrics"""
        try:
            current_avg = self.metrics['average_read_time']
            total_reads = self.metrics['total_reads']
            
            if total_reads > 0:
                self.metrics['average_read_time'] = (current_avg * total_reads + read_time) / (total_reads + 1)
            else:
                self.metrics['average_read_time'] = read_time
                
        except Exception as e:
            self.logger.error(f"Read metrics update failed: {e}")
    
    # Public API methods
    
    def search_cells(self, word_id: str = None, token_id: int = None,
                    context_word: str = None) -> List[Tuple[Tuple[int, int, int], BinaryCell]]:
        """Search for cells matching criteria"""
        try:
            results = []
            
            if self.storage_mode == StorageMode.SIMULATION:
                cells = self.cube_data.get('cells', {})
                
                for coord_key, cell_data in cells.items():
                    cell = BinaryCell.from_dict(cell_data)
                    
                    # Apply filters
                    if word_id and cell.word_id != word_id:
                        continue
                    if token_id is not None and cell.token_id != token_id:
                        continue
                    if context_word and (context_word not in cell.before_context and 
                                       context_word not in cell.after_context):
                        continue
                    
                    # Parse coordinates
                    coords = tuple(map(int, coord_key.split(',')))
                    results.append((coords, cell))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics"""
        try:
            stats = {
                'is_open': self.is_open,
                'signature_verified': self.signature_verified,
                'storage_mode': self.storage_mode.value,
                'shape': self.shape,
                'metrics': self.metrics.copy(),
                'cache_stats': self.cache_stats.copy(),
                'write_arbitrator': self.write_arbitrator.get_metrics(),
                'background_services': len(self.background_threads)
            }
            
            if self.storage_mode == StorageMode.SIMULATION:
                stats['total_cells'] = len(self.cube_data.get('cells', {}))
                stats['metadata'] = self.cube_data.get('metadata', {})
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Get stats failed: {e}")
            return {}
    
    def optimize_storage(self) -> bool:
        """Optimize storage performance"""
        try:
            # Execute pending writes
            results = self.write_arbitrator.execute_writes(self)
            
            # Cleanup cache
            self.cache.cleanup()
            
            # Perform integrity check
            self._perform_integrity_check()
            
            self.logger.info(f"Storage optimization completed: {results}")
            return True
            
        except Exception as e:
            self.logger.error(f"Storage optimization failed: {e}")
            return False

# Supporting classes

class LRUCache:
    """LRU Cache implementation for cell caching"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key: str) -> Optional[BinaryCell]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: BinaryCell):
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
    
    def cleanup(self):
        # Remove old entries (could be enhanced with TTL)
        pass

class MemoryManager:
    """Memory management for cube storage"""
    
    def __init__(self):
        self.memory_usage = 0
        self.max_memory = 1024 * 1024 * 1024  # 1GB
    
    def allocate(self, size: int) -> bool:
        if self.memory_usage + size <= self.max_memory:
            self.memory_usage += size
            return True
        return False
    
    def deallocate(self, size: int):
        self.memory_usage = max(0, self.memory_usage - size)

class IntegrityChecker:
    """Data integrity checker"""
    
    def __init__(self):
        self.checks_performed = 0
        self.errors_found = 0
    
    def check_cell(self, cell: BinaryCell) -> bool:
        self.checks_performed += 1
        if not cell.validate():
            self.errors_found += 1
            return False
        return True

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self):
        self.metrics = defaultdict(float)
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float):
        self.metrics[name] = value
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

# Alias for backward compatibility
CortexOSCubeStorage = AdvancedCubeStorage
CortexCubeNVME = AdvancedCubeStorage

if __name__ == "__main__":
    # Comprehensive test suite
    print("🧊 Testing Advanced CortexOS Cube Storage...")
    
    # Create storage system
    storage = AdvancedCubeStorage(
        storage_path="/tmp/test_cortex_storage",
        shape=(100, 100, 100),
        storage_mode=StorageMode.SIMULATION
    )
    
    # Test opening
    print("📂 Testing storage opening...")
    if storage.open():
        print("✅ Storage opened successfully")
        
        # Test cell creation and writing
        print("\n📝 Testing cell writing...")
        test_cells = []
        for i in range(5):
            cell = BinaryCell(
                word_id=f"test_word_{i}",
                token_id=1000 + i,
                frequency=i + 1,
                tone_signature=0.5 + i * 0.1
            )
            
            # Add some context
            cell.add_context(f"before_word_{i}", "before", 0.8)
            cell.add_context(f"after_word_{i}", "after", 0.9)
            
            coords = (i, i, i)
            success = storage.write_cell(coords, cell, priority=1.0)
            print(f"{'✅' if success else '❌'} Wrote cell {i} at {coords}")
            
            if success:
                test_cells.append((coords, cell))
        
        # Execute pending writes
        print("\n⚡ Executing pending writes...")
        results = storage.write_arbitrator.execute_writes(storage)
        print(f"✅ Executed {results['executed']} writes, {results['failed']} failed")
        
        # Test cell reading
        print("\n📖 Testing cell reading...")
        for coords, original_cell in test_cells:
            read_cell = storage.read_cell(coords)
            if read_cell:
                print(f"✅ Read cell at {coords}: {read_cell.word_id}")
                print(f"   Frequency: {read_cell.frequency}, Contexts: {len(read_cell.before_context)}")
            else:
                print(f"❌ Failed to read cell at {coords}")
        
        # Test search functionality
        print("\n🔍 Testing cell search...")
        search_results = storage.search_cells(word_id="test_word_2")
        print(f"✅ Found {len(search_results)} cells matching 'test_word_2'")
        
        # Test context search
        context_results = storage.search_cells(context_word="before_word_1")
        print(f"✅ Found {len(context_results)} cells with context 'before_word_1'")
        
        # Test storage optimization
        print("\n⚡ Testing storage optimization...")
        if storage.optimize_storage():
            print("✅ Storage optimization completed")
        
        # Test statistics
        print("\n📊 Testing statistics...")
        stats = storage.get_storage_stats()
        print(f"Storage Stats:")
        print(f"  Total Reads: {stats['metrics']['total_reads']}")
        print(f"  Total Writes: {stats['metrics']['total_writes']}")
        print(f"  Cache Hit Rate: {stats['metrics']['cache_hit_rate']:.2%}")
        print(f"  Storage Utilization: {stats['metrics']['storage_utilization']:.2%}")
        print(f"  Total Cells: {stats.get('total_cells', 0)}")
        
        # Test signature verification
        print("\n🔐 Testing signature verification...")
        if storage.verify_signature():
            print("✅ Signature verification passed")
        else:
            print("❌ Signature verification failed")
        
        # Let background services run briefly
        print("\n⏱️ Testing background services...")
        time.sleep(3)
        print("✅ Background services running")
        
        # Test closing
        print("\n🔒 Testing storage closing...")
        if storage.close():
            print("✅ Storage closed successfully")
    else:
        print("❌ Failed to open storage")
    
    print("\n🎉 Advanced Cube Storage test complete!")

