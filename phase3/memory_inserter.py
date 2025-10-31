"""
phase3/memory_inserter.py - CortexOS Memory Insertion Engine
Controls memory insertion operations with trust score gating, retry logic, and robust fallback mechanisms.
"""

import time
import threading
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Path placeholders
MEMORY_DATA_DIR = "{PATH_MEMORY_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
BACKUP_DIR = "{PATH_BACKUP_DIR}"

class MemoryInserter:
    """
    Controls memory insertion operations with trust score gating, retry logic, and robust fallback mechanisms.
    
    Implements comprehensive memory management with validation, coherence checking,
    and full Agharmonic Law compliance for stable cognitive operations.
    """
    
    def __init__(self, trust_threshold: float = 0.65, coherence_threshold: float = 0.75,
                 max_retries: int = 3, retry_backoff: float = 1.5):
        """
        Initialize the Memory Insertion Engine.
        
        Args:
            trust_threshold: Minimum trust score for memory acceptance
            coherence_threshold: Minimum coherence score for memory acceptance
            max_retries: Maximum retry attempts for failed insertions
            retry_backoff: Backoff time between retries (seconds)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Initializing MemoryInserter...")
        
        # Configuration parameters
        self.trust_threshold = trust_threshold
        self.coherence_threshold = coherence_threshold
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        
        # Memory storage and tracking
        self.memory_registry = {}
        self.insertion_queue = deque()
        self.failed_insertions = {}
        self.memory_clusters = defaultdict(list)
        
        # Trust and validation systems
        self.trust_scores = {}
        self.coherence_scores = {}
        self.validation_cache = {}
        
        # Performance metrics
        self.metrics = {
            "total_attempted": 0,
            "successful": 0,
            "rejected_trust": 0,
            "rejected_coherence": 0,
            "failed_after_retry": 0,
            "recovered_by_retry": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Memory insertion statistics
        self.insertion_stats = {
            "avg_trust_score": 0.0,
            "avg_coherence_score": 0.0,
            "avg_insertion_time": 0.0,
            "memory_count": 0,
            "cluster_count": 0
        }
        
        # Temporal management
        self.last_sync = time.time()
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
        self.memory_retention_time = 86400  # 24 hours
        
        # Threading and safety
        self.lock = threading.Lock()
        self.active = True
        
        # Component references
        self.cognitive_bridge = None
        self.sync_manager = None
        self.resonance_monitor = None
        
        # Fallback and health monitoring
        self.fallback_mode = False
        self.fallback_level = 0
        self.system_health = 1.0
        self.error_count = 0
        self.success_count = 0
        
        # Memory validation rules
        self.validation_rules = {
            "min_vector_length": 4,
            "max_vector_length": 1024,
            "min_trust_score": 0.0,
            "max_trust_score": 1.0,
            "min_coherence_score": 0.0,
            "max_coherence_score": 1.0,
            "required_fields": ["id", "harmonics_vector", "trust_score", "coherence_score"]
        }
        
        # Memory clustering parameters
        self.clustering_threshold = 0.8
        self.max_cluster_size = 50
        self.cluster_merge_threshold = 0.9
        
        # Agharmonic Law parameters
        self.input_frequency_range = [0.4, 1.1]
        self.output_phase_alignment = 0.1
        self.resonance_threshold = 0.7
        
        self.logger.info("✅ MemoryInserter initialized")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def insert_memory(self, memory_data: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """
        Insert memory with comprehensive validation and retry logic.
        
        Args:
            memory_data: Memory data to insert
            force: Force insertion bypassing some validation checks
            
        Returns:
            Dictionary with insertion result and metadata
        """
        try:
            start_time = time.time()
            
            with self.lock:
                self.metrics["total_attempted"] += 1
                
                # Validate interface contract
                if not force and not self._validate_interface_contract(memory_data):
                    return {
                        'success': False,
                        'reason': 'interface_contract_validation_failed',
                        'memory_id': memory_data.get('id', 'unknown'),
                        'timestamp': time.time()
                    }
                    
                memory_id = memory_data.get('id', f"mem_{int(time.time() * 1000)}")
                
                # Check if memory already exists
                if memory_id in self.memory_registry:
                    return {
                        'success': False,
                        'reason': 'memory_already_exists',
                        'memory_id': memory_id,
                        'timestamp': time.time()
                    }
                    
                # Extract and validate scores
                trust_score = memory_data.get('trust_score', 0.0)
                coherence_score = memory_data.get('coherence_score', 0.0)
                
                # Trust score validation
                if not force and trust_score < self.trust_threshold:
                    self.metrics["rejected_trust"] += 1
                    return {
                        'success': False,
                        'reason': 'trust_score_too_low',
                        'trust_score': trust_score,
                        'threshold': self.trust_threshold,
                        'memory_id': memory_id,
                        'timestamp': time.time()
                    }
                    
                # Coherence score validation
                if not force and coherence_score < self.coherence_threshold:
                    self.metrics["rejected_coherence"] += 1
                    return {
                        'success': False,
                        'reason': 'coherence_score_too_low',
                        'coherence_score': coherence_score,
                        'threshold': self.coherence_threshold,
                        'memory_id': memory_id,
                        'timestamp': time.time()
                    }
                    
                # Attempt insertion with retry logic
                insertion_result = self._attempt_insertion_with_retry(memory_data, memory_id)
                
                if insertion_result['success']:
                    # Update statistics
                    self.metrics["successful"] += 1
                    self.success_count += 1
                    
                    # Update insertion stats
                    self._update_insertion_stats(trust_score, coherence_score, 
                                               time.time() - start_time)
                    
                    # Perform memory clustering
                    self._cluster_memory(memory_id, memory_data)
                    
                    # Log successful insertion
                    self.logger.debug(f"Memory {memory_id} inserted successfully")
                    
                else:
                    self.error_count += 1
                    
                return insertion_result
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Memory insertion failed: {e}")
            return {
                'success': False,
                'reason': 'insertion_exception',
                'error': str(e),
                'memory_id': memory_data.get('id', 'unknown'),
                'timestamp': time.time()
            }
            
    def batch_insert_memories(self, memory_batch: List[Dict[str, Any]], 
                            force: bool = False) -> Dict[str, Any]:
        """
        Insert multiple memories in batch with optimized processing.
        
        Args:
            memory_batch: List of memory data dictionaries
            force: Force insertion bypassing some validation checks
            
        Returns:
            Batch insertion results with detailed statistics
        """
        try:
            start_time = time.time()
            
            batch_results = {
                'total_memories': len(memory_batch),
                'successful': 0,
                'failed': 0,
                'results': [],
                'processing_time': 0.0,
                'timestamp': time.time()
            }
            
            # Process each memory in the batch
            for memory_data in memory_batch:
                result = self.insert_memory(memory_data, force)
                batch_results['results'].append(result)
                
                if result['success']:
                    batch_results['successful'] += 1
                else:
                    batch_results['failed'] += 1
                    
            batch_results['processing_time'] = time.time() - start_time
            batch_results['success_rate'] = (
                batch_results['successful'] / batch_results['total_memories']
                if batch_results['total_memories'] > 0 else 0.0
            )
            
            self.logger.info(f"Batch insertion completed: {batch_results['successful']}/{batch_results['total_memories']} successful")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch memory insertion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            
    def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory by ID with metadata.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory data with metadata or None if not found
        """
        try:
            with self.lock:
                if memory_id in self.memory_registry:
                    memory_data = self.memory_registry[memory_id].copy()
                    
                    # Add retrieval metadata
                    memory_data['retrieved_at'] = time.time()
                    memory_data['retrieval_count'] = memory_data.get('retrieval_count', 0) + 1
                    
                    # Update retrieval count in registry
                    self.memory_registry[memory_id]['retrieval_count'] = memory_data['retrieval_count']
                    
                    return memory_data
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Memory retrieval failed for {memory_id}: {e}")
            return None
            
    def search_memories(self, query_vector: Union[List, np.ndarray], 
                       similarity_threshold: float = 0.7, 
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories by similarity to query vector.
        
        Args:
            query_vector: Query vector for similarity search
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of similar memories with similarity scores
        """
        try:
            query_array = np.array(query_vector, dtype=np.float32)
            similar_memories = []
            
            with self.lock:
                for memory_id, memory_data in self.memory_registry.items():
                    try:
                        memory_vector = np.array(memory_data['harmonics_vector'], dtype=np.float32)
                        
                        # Calculate similarity
                        similarity = self._calculate_similarity(query_array, memory_vector)
                        
                        if similarity >= similarity_threshold:
                            similar_memories.append({
                                'memory_id': memory_id,
                                'similarity': similarity,
                                'memory_data': memory_data.copy(),
                                'trust_score': memory_data.get('trust_score', 0.0),
                                'coherence_score': memory_data.get('coherence_score', 0.0)
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing memory {memory_id} in search: {e}")
                        continue
                        
            # Sort by similarity and limit results
            similar_memories.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_memories[:max_results]
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []
            
    def update_memory_trust(self, memory_id: str, new_trust_score: float) -> bool:
        """
        Update trust score for an existing memory.
        
        Args:
            memory_id: Unique identifier for the memory
            new_trust_score: New trust score (0.0 to 1.0)
            
        Returns:
            True if update successful
        """
        try:
            if not (0.0 <= new_trust_score <= 1.0):
                self.logger.warning(f"Invalid trust score: {new_trust_score}")
                return False
                
            with self.lock:
                if memory_id in self.memory_registry:
                    old_score = self.memory_registry[memory_id].get('trust_score', 0.0)
                    self.memory_registry[memory_id]['trust_score'] = new_trust_score
                    self.memory_registry[memory_id]['trust_updated_at'] = time.time()
                    
                    self.logger.debug(f"Trust score updated for {memory_id}: {old_score} -> {new_trust_score}")
                    return True
                else:
                    self.logger.warning(f"Memory {memory_id} not found for trust update")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Trust score update failed for {memory_id}: {e}")
            return False
            
    def delete_memory(self, memory_id: str, backup: bool = True) -> bool:
        """
        Delete memory with optional backup.
        
        Args:
            memory_id: Unique identifier for the memory
            backup: Whether to create backup before deletion
            
        Returns:
            True if deletion successful
        """
        try:
            with self.lock:
                if memory_id not in self.memory_registry:
                    self.logger.warning(f"Memory {memory_id} not found for deletion")
                    return False
                    
                memory_data = self.memory_registry[memory_id]
                
                # Create backup if requested
                if backup:
                    backup_data = {
                        'memory_id': memory_id,
                        'memory_data': memory_data.copy(),
                        'deleted_at': time.time(),
                        'deletion_reason': 'manual_deletion'
                    }
                    
                    # Store backup (in production, this would go to persistent storage)
                    backup_key = f"backup_{memory_id}_{int(time.time())}"
                    # self._store_backup(backup_key, backup_data)  # Placeholder
                    
                # Remove from registry
                del self.memory_registry[memory_id]
                
                # Remove from clusters
                self._remove_from_clusters(memory_id)
                
                # Update statistics
                self.insertion_stats['memory_count'] = len(self.memory_registry)
                
                self.logger.debug(f"Memory {memory_id} deleted successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Memory deletion failed for {memory_id}: {e}")
            return False
            
    def cleanup_expired_memories(self) -> Dict[str, Any]:
        """
        Clean up expired memories based on retention time.
        
        Returns:
            Cleanup statistics
        """
        try:
            current_time = time.time()
            
            # Check if cleanup is needed
            if current_time - self.last_cleanup < self.cleanup_interval:
                return {'status': 'cleanup_not_needed'}
                
            cleanup_stats = {
                'expired_count': 0,
                'low_trust_count': 0,
                'cluster_merges': 0,
                'total_cleaned': 0
            }
            
            with self.lock:
                expired_memories = []
                
                # Find expired memories
                for memory_id, memory_data in self.memory_registry.items():
                    memory_age = current_time - memory_data.get('inserted_at', current_time)
                    
                    # Check expiration
                    if memory_age > self.memory_retention_time:
                        expired_memories.append(memory_id)
                        cleanup_stats['expired_count'] += 1
                        
                    # Check low trust scores
                    elif memory_data.get('trust_score', 1.0) < 0.3:
                        expired_memories.append(memory_id)
                        cleanup_stats['low_trust_count'] += 1
                        
                # Remove expired memories
                for memory_id in expired_memories:
                    self.delete_memory(memory_id, backup=True)
                    
                cleanup_stats['total_cleaned'] = len(expired_memories)
                
                # Merge similar clusters
                cluster_merges = self._merge_similar_clusters()
                cleanup_stats['cluster_merges'] = cluster_merges
                
                # Update statistics
                self.insertion_stats['memory_count'] = len(self.memory_registry)
                self.insertion_stats['cluster_count'] = len(self.memory_clusters)
                
                self.last_cleanup = current_time
                
            self.logger.info(f"Memory cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return {'status': 'cleanup_failed', 'error': str(e)}
            
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory insertion statistics"""
        try:
            with self.lock:
                current_time = time.time()
                
                # Calculate health metrics
                total_ops = self.success_count + self.error_count
                success_rate = self.success_count / max(1, total_ops)
                
                # Memory distribution by trust/coherence scores
                trust_distribution = {'high': 0, 'medium': 0, 'low': 0}
                coherence_distribution = {'high': 0, 'medium': 0, 'low': 0}
                
                for memory_data in self.memory_registry.values():
                    trust = memory_data.get('trust_score', 0.0)
                    coherence = memory_data.get('coherence_score', 0.0)
                    
                    # Trust distribution
                    if trust >= 0.8:
                        trust_distribution['high'] += 1
                    elif trust >= 0.5:
                        trust_distribution['medium'] += 1
                    else:
                        trust_distribution['low'] += 1
                        
                    # Coherence distribution
                    if coherence >= 0.8:
                        coherence_distribution['high'] += 1
                    elif coherence >= 0.5:
                        coherence_distribution['medium'] += 1
                    else:
                        coherence_distribution['low'] += 1
                        
                return {
                    'timestamp': current_time,
                    'memory_counts': {
                        'total_memories': len(self.memory_registry),
                        'total_clusters': len(self.memory_clusters),
                        'failed_insertions': len(self.failed_insertions)
                    },
                    'insertion_metrics': self.metrics.copy(),
                    'insertion_stats': self.insertion_stats.copy(),
                    'performance': {
                        'success_rate': success_rate,
                        'system_health': self.system_health,
                        'fallback_mode': self.fallback_mode,
                        'fallback_level': self.fallback_level
                    },
                    'distributions': {
                        'trust_scores': trust_distribution,
                        'coherence_scores': coherence_distribution
                    },
                    'thresholds': {
                        'trust_threshold': self.trust_threshold,
                        'coherence_threshold': self.coherence_threshold,
                        'clustering_threshold': self.clustering_threshold
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}
            
    # Private helper methods
    def _validate_interface_contract(self, memory_data: Dict[str, Any]) -> bool:
        """Validate memory data against interface contract"""
        try:
            # Check required fields
            for field in self.validation_rules["required_fields"]:
                if field not in memory_data:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
                    
            # Validate harmonics vector
            harmonics_vector = memory_data.get('harmonics_vector', [])
            if not isinstance(harmonics_vector, (list, tuple, np.ndarray)):
                self.logger.warning("harmonics_vector must be a list, tuple, or numpy array")
                return False
                
            vector_length = len(harmonics_vector)
            if not (self.validation_rules["min_vector_length"] <= vector_length <= 
                   self.validation_rules["max_vector_length"]):
                self.logger.warning(f"Invalid vector length: {vector_length}")
                return False
                
            # Validate trust score
            trust_score = memory_data.get('trust_score', 0.0)
            if not (self.validation_rules["min_trust_score"] <= trust_score <= 
                   self.validation_rules["max_trust_score"]):
                self.logger.warning(f"Invalid trust score: {trust_score}")
                return False
                
            # Validate coherence score
            coherence_score = memory_data.get('coherence_score', 0.0)
            if not (self.validation_rules["min_coherence_score"] <= coherence_score <= 
                   self.validation_rules["max_coherence_score"]):
                self.logger.warning(f"Invalid coherence score: {coherence_score}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Interface contract validation failed: {e}")
            return False
            
    def _attempt_insertion_with_retry(self, memory_data: Dict[str, Any], 
                                    memory_id: str) -> Dict[str, Any]:
        """Attempt memory insertion with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Perform the actual insertion
                insertion_success = self._perform_memory_insertion(memory_data, memory_id)
                
                if insertion_success:
                    if attempt > 0:
                        self.metrics["recovered_by_retry"] += 1
                        
                    return {
                        'success': True,
                        'memory_id': memory_id,
                        'attempts': attempt + 1,
                        'timestamp': time.time()
                    }
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Insertion attempt {attempt + 1} failed for {memory_id}: {e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    
        # All attempts failed
        self.metrics["failed_after_retry"] += 1
        self.failed_insertions[memory_id] = {
            'memory_data': memory_data,
            'last_error': str(last_error),
            'failed_at': time.time(),
            'attempts': self.max_retries + 1
        }
        
        return {
            'success': False,
            'reason': 'insertion_failed_after_retry',
            'memory_id': memory_id,
            'attempts': self.max_retries + 1,
            'last_error': str(last_error),
            'timestamp': time.time()
        }
        
    def _perform_memory_insertion(self, memory_data: Dict[str, Any], memory_id: str) -> bool:
        """Perform the actual memory insertion"""
        try:
            # Prepare memory entry
            memory_entry = memory_data.copy()
            memory_entry['inserted_at'] = time.time()
            memory_entry['retrieval_count'] = 0
            memory_entry['last_accessed'] = time.time()
            
            # Store in registry
            self.memory_registry[memory_id] = memory_entry
            
            # Store trust and coherence scores
            self.trust_scores[memory_id] = memory_data.get('trust_score', 0.0)
            self.coherence_scores[memory_id] = memory_data.get('coherence_score', 0.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory insertion failed for {memory_id}: {e}")
            return False
            
    def _update_insertion_stats(self, trust_score: float, coherence_score: float, 
                              processing_time: float):
        """Update insertion statistics"""
        try:
            memory_count = len(self.memory_registry)
            
            # Update averages
            if memory_count > 1:
                self.insertion_stats['avg_trust_score'] = (
                    (self.insertion_stats['avg_trust_score'] * (memory_count - 1) + trust_score) / 
                    memory_count
                )
                self.insertion_stats['avg_coherence_score'] = (
                    (self.insertion_stats['avg_coherence_score'] * (memory_count - 1) + coherence_score) / 
                    memory_count
                )
                self.insertion_stats['avg_insertion_time'] = (
                    (self.insertion_stats['avg_insertion_time'] * (memory_count - 1) + processing_time) / 
                    memory_count
                )
            else:
                self.insertion_stats['avg_trust_score'] = trust_score
                self.insertion_stats['avg_coherence_score'] = coherence_score
                self.insertion_stats['avg_insertion_time'] = processing_time
                
            self.insertion_stats['memory_count'] = memory_count
            
        except Exception as e:
            self.logger.error(f"Statistics update failed: {e}")
            
    def _cluster_memory(self, memory_id: str, memory_data: Dict[str, Any]):
        """Cluster memory based on similarity"""
        try:
            memory_vector = np.array(memory_data['harmonics_vector'], dtype=np.float32)
            best_cluster = None
            best_similarity = 0.0
            
            # Find best matching cluster
            for cluster_id, cluster_memories in self.memory_clusters.items():
                if len(cluster_memories) >= self.max_cluster_size:
                    continue
                    
                # Calculate average similarity to cluster
                cluster_similarities = []
                for cluster_memory_id in cluster_memories:
                    if cluster_memory_id in self.memory_registry:
                        cluster_vector = np.array(
                            self.memory_registry[cluster_memory_id]['harmonics_vector'], 
                            dtype=np.float32
                        )
                        similarity = self._calculate_similarity(memory_vector, cluster_vector)
                        cluster_similarities.append(similarity)
                        
                if cluster_similarities:
                    avg_similarity = np.mean(cluster_similarities)
                    if avg_similarity > best_similarity and avg_similarity >= self.clustering_threshold:
                        best_similarity = avg_similarity
                        best_cluster = cluster_id
                        
            # Add to best cluster or create new one
            if best_cluster:
                self.memory_clusters[best_cluster].append(memory_id)
            else:
                # Create new cluster
                new_cluster_id = f"cluster_{len(self.memory_clusters)}"
                self.memory_clusters[new_cluster_id] = [memory_id]
                
            self.insertion_stats['cluster_count'] = len(self.memory_clusters)
            
        except Exception as e:
            self.logger.error(f"Memory clustering failed for {memory_id}: {e}")
            
    def _remove_from_clusters(self, memory_id: str):
        """Remove memory from all clusters"""
        try:
            clusters_to_remove = []
            
            for cluster_id, cluster_memories in self.memory_clusters.items():
                if memory_id in cluster_memories:
                    cluster_memories.remove(memory_id)
                    
                    # Remove empty clusters
                    if not cluster_memories:
                        clusters_to_remove.append(cluster_id)
                        
            for cluster_id in clusters_to_remove:
                del self.memory_clusters[cluster_id]
                
        except Exception as e:
            self.logger.error(f"Cluster removal failed for {memory_id}: {e}")
            
    def _merge_similar_clusters(self) -> int:
        """Merge similar clusters to optimize organization"""
        try:
            merge_count = 0
            cluster_ids = list(self.memory_clusters.keys())
            
            for i, cluster_id1 in enumerate(cluster_ids):
                if cluster_id1 not in self.memory_clusters:
                    continue
                    
                for cluster_id2 in cluster_ids[i+1:]:
                    if cluster_id2 not in self.memory_clusters:
                        continue
                        
                    # Calculate inter-cluster similarity
                    similarity = self._calculate_cluster_similarity(cluster_id1, cluster_id2)
                    
                    if similarity >= self.cluster_merge_threshold:
                        # Merge clusters
                        self.memory_clusters[cluster_id1].extend(self.memory_clusters[cluster_id2])
                        del self.memory_clusters[cluster_id2]
                        merge_count += 1
                        
            return merge_count
            
        except Exception as e:
            self.logger.error(f"Cluster merging failed: {e}")
            return 0
            
    def _calculate_cluster_similarity(self, cluster_id1: str, cluster_id2: str) -> float:
        """Calculate similarity between two clusters"""
        try:
            cluster1_memories = self.memory_clusters[cluster_id1]
            cluster2_memories = self.memory_clusters[cluster_id2]
            
            similarities = []
            
            for memory_id1 in cluster1_memories:
                if memory_id1 not in self.memory_registry:
                    continue
                    
                vector1 = np.array(self.memory_registry[memory_id1]['harmonics_vector'], dtype=np.float32)
                
                for memory_id2 in cluster2_memories:
                    if memory_id2 not in self.memory_registry:
                        continue
                        
                    vector2 = np.array(self.memory_registry[memory_id2]['harmonics_vector'], dtype=np.float32)
                    similarity = self._calculate_similarity(vector1, vector2)
                    similarities.append(similarity)
                    
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Cluster similarity calculation failed: {e}")
            return 0.0
            
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate similarity between two vectors"""
        try:
            # Ensure vectors are the same length
            min_len = min(len(vector1), len(vector2))
            v1 = vector1[:min_len]
            v2 = vector2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
            
    # Agharmonic Law Compliance Methods
    def harmonic_signature(self) -> Dict[str, Any]:
        """Establish frequency compatibility parameters"""
        return {
            "module": "memory_inserter",
            "input_frequency_range": self.input_frequency_range,
            "output_phase_alignment": self.output_phase_alignment,
            "resonance_threshold": self.resonance_threshold,
            "harmonic_modes": ["memory_insertion", "trust_validation", "coherence_checking"],
            "compatible_modules": ["cognitive_bridge", "resonance_monitor", "neuroengine"]
        }
        
    def interface_contract(self) -> Dict[str, List[str]]:
        """Define interface contract for external modules"""
        return {
            "inputs": ["memory_data", "trust_threshold", "coherence_threshold"],
            "outputs": ["insertion_result", "memory_statistics", "cluster_information"],
            "methods": [
                "insert_memory",
                "batch_insert_memories", 
                "retrieve_memory",
                "search_memories",
                "update_memory_trust",
                "delete_memory"
            ]
        }
        
    def cognitive_energy_flow(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signals for memory insertion operations"""
        try:
            if signal.size == 0:
                return np.array([])
                
            # Apply trust-based filtering
            trust_factor = np.mean([score for score in self.trust_scores.values()]) if self.trust_scores else 0.5
            
            # Modulate signal based on system trust
            modulated_signal = signal * (0.5 + 0.5 * trust_factor)
            
            # Normalize to [0.05, 0.95] range
            signal_min = modulated_signal.min()
            signal_max = modulated_signal.max()
            
            if signal_max - signal_min == 0:
                return np.full_like(modulated_signal, 0.5)
                
            normalized = (modulated_signal - signal_min) / (signal_max - signal_min)
            return normalized * 0.9 + 0.05
            
        except Exception as e:
            self.logger.error(f"Cognitive energy flow failed: {e}")
            return signal
            
    def sync_clock(self, global_clock: Any = None) -> bool:
        """Connect to master temporal framework"""
        try:
            if global_clock and hasattr(global_clock, 'get_time'):
                synchronized_time = global_clock.get_time()
                self.last_sync = synchronized_time
                return True
            else:
                self.last_sync = time.time()
                return True
                
        except Exception as e:
            self.logger.error(f"Clock synchronization failed: {e}")
            return False
            
    def self_regulate(self) -> Dict[str, Any]:
        """Perform self-regulation and internal monitoring"""
        try:
            # Perform automatic cleanup
            cleanup_result = self.cleanup_expired_memories()
            
            # Update system health
            total_ops = self.success_count + self.error_count
            if total_ops > 0:
                self.system_health = self.success_count / total_ops
            else:
                self.system_health = 1.0
                
            # Adjust thresholds based on performance
            if self.system_health < 0.8:
                # Lower thresholds to improve success rate
                self.trust_threshold = max(0.3, self.trust_threshold - 0.05)
                self.coherence_threshold = max(0.3, self.coherence_threshold - 0.05)
                
            return {
                'status': 'regulation_completed',
                'system_health': self.system_health,
                'cleanup_result': cleanup_result,
                'adjusted_thresholds': {
                    'trust_threshold': self.trust_threshold,
                    'coherence_threshold': self.coherence_threshold
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Self-regulation failed: {e}")
            return {'status': 'regulation_failed', 'error': str(e)}
            
    def graceful_fallback(self) -> Dict[str, Any]:
        """Implement graceful fallback for memory insertion failures"""
        try:
            self.fallback_mode = True
            self.fallback_level += 1
            
            fallback_actions = []
            
            if self.fallback_level == 1:
                # Level 1: Reduce validation strictness
                self.trust_threshold *= 0.8
                self.coherence_threshold *= 0.8
                fallback_actions.append('reduced_validation_strictness')
                
            elif self.fallback_level == 2:
                # Level 2: Disable clustering
                self.memory_clusters.clear()
                fallback_actions.append('disabled_clustering')
                
            elif self.fallback_level >= 3:
                # Level 3: Minimal operation mode
                self.trust_threshold = 0.1
                self.coherence_threshold = 0.1
                self.max_retries = 1
                fallback_actions.append('minimal_operation_mode')
                
            return {
                'status': 'fallback_activated',
                'level': self.fallback_level,
                'actions': fallback_actions,
                'reduced_capabilities': [
                    'strict_validation',
                    'memory_clustering',
                    'complex_retry_logic'
                ],
                'maintained_capabilities': [
                    'basic_memory_insertion',
                    'memory_retrieval',
                    'simple_search'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Graceful fallback failed: {e}")
            return {'status': 'fallback_failed', 'error': str(e)}
            
    def resonance_chain_validator(self, chain_data: Dict = None) -> bool:
        """Validate memory insertion chain integrity"""
        try:
            if chain_data is None:
                # Validate internal state
                if not self.memory_registry:
                    return True  # Empty state is valid
                    
                # Check memory data integrity
                for memory_id, memory_data in self.memory_registry.items():
                    if not self._validate_interface_contract(memory_data):
                        return False
                        
                # Check cluster consistency
                for cluster_id, cluster_memories in self.memory_clusters.items():
                    for memory_id in cluster_memories:
                        if memory_id not in self.memory_registry:
                            return False
                            
                return True
            else:
                # Validate external chain data
                required_fields = ['memories', 'clusters']
                return all(field in chain_data for field in required_fields)
                
        except Exception as e:
            self.logger.error(f"Resonance chain validation failed: {e}")
            return False
            
    # Public Interface Methods
    def set_thresholds(self, trust_threshold: float = None, 
                      coherence_threshold: float = None) -> bool:
        """Set validation thresholds"""
        try:
            if trust_threshold is not None:
                if 0.0 <= trust_threshold <= 1.0:
                    self.trust_threshold = trust_threshold
                else:
                    return False
                    
            if coherence_threshold is not None:
                if 0.0 <= coherence_threshold <= 1.0:
                    self.coherence_threshold = coherence_threshold
                else:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set thresholds: {e}")
            return False
            
    def reset_system(self) -> bool:
        """Reset the entire memory insertion system"""
        try:
            with self.lock:
                self.memory_registry.clear()
                self.insertion_queue.clear()
                self.failed_insertions.clear()
                self.memory_clusters.clear()
                self.trust_scores.clear()
                self.coherence_scores.clear()
                self.validation_cache.clear()
                
                # Reset metrics
                for key in self.metrics:
                    self.metrics[key] = 0
                    
                # Reset statistics
                self.insertion_stats = {
                    "avg_trust_score": 0.0,
                    "avg_coherence_score": 0.0,
                    "avg_insertion_time": 0.0,
                    "memory_count": 0,
                    "cluster_count": 0
                }
                
                self.success_count = 0
                self.error_count = 0
                self.fallback_mode = False
                self.fallback_level = 0
                self.system_health = 1.0
                
                self.logger.info("Memory insertion system reset successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reset system: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS Memory Inserter
    print("💾 Testing CortexOS Memory Inserter...")
    
    inserter = CortexOSMemoryInserter()
    
    # Test inserter startup
    if inserter.start():
        print("✅ Memory Inserter started successfully")
    
    # Test memory insertion with various data types
    test_memories = [
        {
            "content": "CortexOS is a neural cognitive architecture",
            "memory_type": "factual",
            "context": "system_knowledge",
            "importance": 0.9,
            "source": "documentation"
        },
        {
            "content": {"pattern": [1, 0, 1, 1, 0], "frequency": 440.0},
            "memory_type": "pattern",
            "context": "neural_patterns",
            "importance": 0.8,
            "source": "pattern_recognition"
        },
        {
            "content": "User prefers analytical processing mode",
            "memory_type": "preference",
            "context": "user_profile",
            "importance": 0.7,
            "source": "user_interaction"
        },
        {
            "content": "Resonance field stability improved after optimization",
            "memory_type": "experience",
            "context": "system_learning",
            "importance": 0.6,
            "source": "system_monitor"
        }
    ]
    
    inserted_ids = []
    for i, memory in enumerate(test_memories):
        memory_id = inserter.insert_memory(memory)
        if memory_id:
            inserted_ids.append(memory_id)
            print(f"✅ Inserted memory {i+1}: {memory['memory_type']} - ID: {memory_id[:8]}...")
    
    # Test memory retrieval
    if inserted_ids:
        retrieved = inserter.retrieve_memory(inserted_ids[0])
        if retrieved:
            print(f"✅ Retrieved memory: {retrieved['memory_type']} with trust score {retrieved['trust_score']:.2f}")
    
    # Test memory search
    search_results = inserter.search_memories("CortexOS", limit=3)
    print(f"✅ Search for 'CortexOS' returned {len(search_results)} results")
    
    # Test clustering
    clusters = inserter.cluster_memories()
    print(f"✅ Memory clustering created {len(clusters)} clusters")
    
    # Test trust score validation
    trust_tests = [
        {"content": "High quality factual information", "expected": "high"},
        {"content": "Uncertain or questionable data", "expected": "medium"},
        {"content": "Low confidence information", "expected": "low"}
    ]
    
    for test in trust_tests:
        trust_score = inserter.calculate_trust_score(test["content"], "test", "validation")
        print(f"✅ Trust score for {test['expected']} quality: {trust_score:.2f}")
    
    # Test memory statistics
    stats = inserter.get_memory_statistics()
    print(f"✅ Memory stats - Total: {stats['total_memories']}, Avg trust: {stats['avg_trust_score']:.2f}")
    
    # Test performance metrics
    performance = inserter.get_performance_metrics()
    print(f"✅ Performance - Success rate: {performance['success_rate']:.2f}, Avg insertion time: {performance['avg_insertion_time']:.3f}s")
    
    # Test memory validation
    validation_result = inserter.validate_memory_integrity()
    print(f"✅ Memory integrity validation: {'Passed' if validation_result else 'Failed'}")
    
    # Test inserter status
    status = inserter.get_status()
    print(f"✅ Inserter status: {status['state']}, Health: {status['system_health']:.2f}")
    
    # Test system reset
    if inserter.reset_system():
        print("✅ Memory insertion system reset successful")
    
    # Shutdown
    inserter.stop()
    print("✅ Memory Inserter stopped")
    
    print("💾 Memory Inserter test complete!")

