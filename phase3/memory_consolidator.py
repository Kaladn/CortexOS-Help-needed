#!/usr/bin/env python3
"""
CortexOS Phase 3: Memory Consolidator
Advanced neural memory consolidation and optimization system
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConsolidationRule:
    """Memory consolidation rule definition"""
    rule_id: str
    name: str
    pattern: Dict[str, Any]
    action: str  # 'merge', 'archive', 'delete', 'compress'
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsolidationTask:
    """Memory consolidation task"""
    task_id: str
    rule: ConsolidationRule
    target_memories: List[str]
    scheduled_time: datetime
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class ConsolidationMetrics:
    """Memory consolidation performance metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    memories_processed: int = 0
    memories_merged: int = 0
    memories_archived: int = 0
    memories_deleted: int = 0
    space_saved: int = 0
    processing_time: float = 0.0

class MemoryCluster:
    """Memory clustering for consolidation"""
    
    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.memories = []
        self.centroid = {}
        self.similarity_threshold = 0.8
        self.last_updated = datetime.now()
        
    def add_memory(self, memory_id: str, content: Dict[str, Any]):
        """Add memory to cluster"""
        self.memories.append({
            'id': memory_id,
            'content': content,
            'added_time': datetime.now()
        })
        self._update_centroid()
        
    def _update_centroid(self):
        """Update cluster centroid"""
        if not self.memories:
            return
            
        # Simple centroid calculation
        all_keys = set()
        for memory in self.memories:
            all_keys.update(memory['content'].keys())
        
        centroid = {}
        for key in all_keys:
            values = [memory['content'].get(key) for memory in self.memories if key in memory['content']]
            if values:
                # For numeric values, use mean; for strings, use most common
                if all(isinstance(v, (int, float)) for v in values):
                    centroid[key] = sum(values) / len(values)
                else:
                    # Most common value
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[str(v)] += 1
                    centroid[key] = max(value_counts.items(), key=lambda x: x[1])[0]
        
        self.centroid = centroid
        self.last_updated = datetime.now()
    
    def calculate_similarity(self, content: Dict[str, Any]) -> float:
        """Calculate similarity to cluster centroid"""
        if not self.centroid:
            return 0.0
            
        common_keys = set(self.centroid.keys()) & set(content.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if str(self.centroid[key]) == str(content[key]):
                matches += 1
        
        return matches / len(common_keys)
    
    def should_merge(self) -> bool:
        """Check if cluster should be merged"""
        return len(self.memories) >= 3 and self._calculate_internal_similarity() > self.similarity_threshold
    
    def _calculate_internal_similarity(self) -> float:
        """Calculate internal cluster similarity"""
        if len(self.memories) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(self.memories)):
            for j in range(i + 1, len(self.memories)):
                sim = self._memory_similarity(
                    self.memories[i]['content'],
                    self.memories[j]['content']
                )
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _memory_similarity(self, content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
        """Calculate similarity between two memories"""
        all_keys = set(content1.keys()) | set(content2.keys())
        if not all_keys:
            return 0.0
        
        matches = 0
        for key in all_keys:
            if key in content1 and key in content2:
                if str(content1[key]) == str(content2[key]):
                    matches += 1
        
        return matches / len(all_keys)

class MemoryConsolidator:
    """Advanced neural memory consolidation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_store = {}
        self.consolidation_rules = {}
        self.clusters = {}
        self.task_queue = asyncio.Queue()
        self.metrics = ConsolidationMetrics()
        self.running = False
        self.worker_tasks = []
        
        # Configuration
        self.worker_count = self.config.get('worker_count', 2)
        self.consolidation_interval = self.config.get('consolidation_interval', 3600)  # 1 hour
        self.max_cluster_size = self.config.get('max_cluster_size', 50)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.archive_age_days = self.config.get('archive_age_days', 30)
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Memory Consolidator initialized")
    
    def _initialize_default_rules(self):
        """Initialize default consolidation rules"""
        # Rule 1: Merge similar memories
        merge_rule = ConsolidationRule(
            rule_id="merge_similar",
            name="Merge Similar Memories",
            pattern={'similarity_threshold': 0.9},
            action="merge",
            priority=1,
            conditions={'min_memories': 2, 'max_age_hours': 24},
            parameters={'preserve_metadata': True}
        )
        self.consolidation_rules[merge_rule.rule_id] = merge_rule
        
        # Rule 2: Archive old memories
        archive_rule = ConsolidationRule(
            rule_id="archive_old",
            name="Archive Old Memories",
            pattern={'age_days': self.archive_age_days},
            action="archive",
            priority=2,
            conditions={'access_count': 0},
            parameters={'compression': True}
        )
        self.consolidation_rules[archive_rule.rule_id] = archive_rule
        
        # Rule 3: Delete duplicate memories
        delete_rule = ConsolidationRule(
            rule_id="delete_duplicates",
            name="Delete Duplicate Memories",
            pattern={'exact_match': True},
            action="delete",
            priority=3,
            conditions={'keep_newest': True},
            parameters={'backup_before_delete': True}
        )
        self.consolidation_rules[delete_rule.rule_id] = delete_rule
    
    async def start(self):
        """Start the memory consolidation system"""
        try:
            self.running = True
            
            # Start worker tasks
            for i in range(self.worker_count):
                task = asyncio.create_task(self._consolidation_worker(f"worker_{i}"))
                self.worker_tasks.append(task)
            
            # Start periodic consolidation
            periodic_task = asyncio.create_task(self._periodic_consolidation())
            self.worker_tasks.append(periodic_task)
            
            logger.info(f"Memory Consolidator started with {self.worker_count} workers")
            
        except Exception as e:
            logger.error(f"Error starting Memory Consolidator: {e}")
            raise
    
    async def stop(self):
        """Stop the memory consolidation system"""
        try:
            self.running = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            logger.info("Memory Consolidator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Memory Consolidator: {e}")
    
    async def add_memory(self, memory_id: str, content: Dict[str, Any], 
                        metadata: Dict[str, Any] = None) -> bool:
        """Add memory to consolidation system"""
        try:
            metadata = metadata or {}
            timestamp = datetime.now()
            
            # Store memory
            self.memory_store[memory_id] = {
                'content': content,
                'metadata': metadata,
                'timestamp': timestamp,
                'access_count': 0,
                'last_accessed': timestamp,
                'consolidated': False
            }
            
            # Add to appropriate cluster
            await self._cluster_memory(memory_id, content)
            
            logger.debug(f"Added memory {memory_id} to consolidation system")
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory {memory_id}: {e}")
            return False
    
    async def _cluster_memory(self, memory_id: str, content: Dict[str, Any]):
        """Add memory to appropriate cluster"""
        try:
            best_cluster = None
            best_similarity = 0.0
            
            # Find best matching cluster
            for cluster in self.clusters.values():
                similarity = cluster.calculate_similarity(content)
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster
            
            # Create new cluster if no good match
            if best_cluster is None:
                cluster_id = f"cluster_{len(self.clusters)}"
                best_cluster = MemoryCluster(cluster_id)
                self.clusters[cluster_id] = best_cluster
            
            # Add memory to cluster
            best_cluster.add_memory(memory_id, content)
            
            # Check if cluster should be consolidated
            if best_cluster.should_merge():
                await self._schedule_consolidation_task(best_cluster)
            
        except Exception as e:
            logger.error(f"Error clustering memory {memory_id}: {e}")
    
    async def _schedule_consolidation_task(self, cluster: MemoryCluster):
        """Schedule consolidation task for cluster"""
        try:
            # Find applicable rule
            rule = self._find_applicable_rule(cluster)
            if not rule:
                return
            
            # Create consolidation task
            task = ConsolidationTask(
                task_id=f"task_{int(time.time())}_{cluster.cluster_id}",
                rule=rule,
                target_memories=[mem['id'] for mem in cluster.memories],
                scheduled_time=datetime.now()
            )
            
            # Add to queue
            await self.task_queue.put(task)
            
            logger.debug(f"Scheduled consolidation task {task.task_id} for cluster {cluster.cluster_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling consolidation task: {e}")
    
    def _find_applicable_rule(self, cluster: MemoryCluster) -> Optional[ConsolidationRule]:
        """Find applicable consolidation rule for cluster"""
        try:
            # Sort rules by priority
            sorted_rules = sorted(self.consolidation_rules.values(), key=lambda r: r.priority)
            
            for rule in sorted_rules:
                if self._rule_matches_cluster(rule, cluster):
                    return rule
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding applicable rule: {e}")
            return None
    
    def _rule_matches_cluster(self, rule: ConsolidationRule, cluster: MemoryCluster) -> bool:
        """Check if rule matches cluster"""
        try:
            # Check memory count condition
            if 'min_memories' in rule.conditions:
                if len(cluster.memories) < rule.conditions['min_memories']:
                    return False
            
            # Check age condition
            if 'max_age_hours' in rule.conditions:
                max_age = timedelta(hours=rule.conditions['max_age_hours'])
                oldest_memory = min(cluster.memories, key=lambda m: m['added_time'])
                if datetime.now() - oldest_memory['added_time'] > max_age:
                    return False
            
            # Check similarity threshold
            if 'similarity_threshold' in rule.pattern:
                if cluster._calculate_internal_similarity() < rule.pattern['similarity_threshold']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rule match: {e}")
            return False
    
    async def _consolidation_worker(self, worker_id: str):
        """Worker task for processing consolidation tasks"""
        logger.info(f"Consolidation worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Process task
                await self._process_consolidation_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in consolidation worker {worker_id}: {e}")
        
        logger.info(f"Consolidation worker {worker_id} stopped")
    
    async def _process_consolidation_task(self, task: ConsolidationTask):
        """Process a single consolidation task"""
        try:
            start_time = time.time()
            task.status = 'running'
            
            logger.info(f"Processing consolidation task {task.task_id} with action '{task.rule.action}'")
            
            if task.rule.action == 'merge':
                result = await self._merge_memories(task)
            elif task.rule.action == 'archive':
                result = await self._archive_memories(task)
            elif task.rule.action == 'delete':
                result = await self._delete_memories(task)
            elif task.rule.action == 'compress':
                result = await self._compress_memories(task)
            else:
                raise ValueError(f"Unknown consolidation action: {task.rule.action}")
            
            # Update task
            task.status = 'completed'
            task.result = result
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.total_tasks += 1
            self.metrics.completed_tasks += 1
            self.metrics.processing_time += processing_time
            self.metrics.memories_processed += len(task.target_memories)
            
            logger.info(f"Completed consolidation task {task.task_id} in {processing_time:.3f}s")
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            self.metrics.failed_tasks += 1
            logger.error(f"Failed consolidation task {task.task_id}: {e}")
    
    async def _merge_memories(self, task: ConsolidationTask) -> Dict[str, Any]:
        """Merge memories according to task"""
        try:
            if len(task.target_memories) < 2:
                return {'action': 'merge', 'merged_count': 0, 'reason': 'insufficient_memories'}
            
            # Get memory contents
            memories_to_merge = []
            for memory_id in task.target_memories:
                if memory_id in self.memory_store:
                    memories_to_merge.append(self.memory_store[memory_id])
            
            if len(memories_to_merge) < 2:
                return {'action': 'merge', 'merged_count': 0, 'reason': 'memories_not_found'}
            
            # Create merged memory
            merged_content = self._merge_memory_contents([m['content'] for m in memories_to_merge])
            merged_metadata = self._merge_memory_metadata([m['metadata'] for m in memories_to_merge])
            
            # Create new merged memory ID
            merged_id = f"merged_{int(time.time())}_{hashlib.md5(''.join(task.target_memories).encode()).hexdigest()[:8]}"
            
            # Store merged memory
            self.memory_store[merged_id] = {
                'content': merged_content,
                'metadata': merged_metadata,
                'timestamp': datetime.now(),
                'access_count': sum(m['access_count'] for m in memories_to_merge),
                'last_accessed': max(m['last_accessed'] for m in memories_to_merge),
                'consolidated': True,
                'source_memories': task.target_memories
            }
            
            # Remove original memories
            for memory_id in task.target_memories:
                if memory_id in self.memory_store:
                    del self.memory_store[memory_id]
            
            self.metrics.memories_merged += len(task.target_memories)
            
            return {
                'action': 'merge',
                'merged_count': len(task.target_memories),
                'new_memory_id': merged_id,
                'space_saved': self._estimate_space_saved(memories_to_merge)
            }
            
        except Exception as e:
            logger.error(f"Error merging memories: {e}")
            raise
    
    async def _archive_memories(self, task: ConsolidationTask) -> Dict[str, Any]:
        """Archive memories according to task"""
        try:
            archived_count = 0
            
            for memory_id in task.target_memories:
                if memory_id in self.memory_store:
                    memory = self.memory_store[memory_id]
                    
                    # Check if memory should be archived
                    age = datetime.now() - memory['timestamp']
                    if age.days >= self.archive_age_days and memory['access_count'] == 0:
                        # Mark as archived
                        memory['archived'] = True
                        memory['archive_date'] = datetime.now()
                        archived_count += 1
            
            self.metrics.memories_archived += archived_count
            
            return {
                'action': 'archive',
                'archived_count': archived_count
            }
            
        except Exception as e:
            logger.error(f"Error archiving memories: {e}")
            raise
    
    async def _delete_memories(self, task: ConsolidationTask) -> Dict[str, Any]:
        """Delete memories according to task"""
        try:
            deleted_count = 0
            
            # Group by content hash to find duplicates
            content_groups = defaultdict(list)
            for memory_id in task.target_memories:
                if memory_id in self.memory_store:
                    memory = self.memory_store[memory_id]
                    content_hash = hashlib.md5(json.dumps(memory['content'], sort_keys=True).encode()).hexdigest()
                    content_groups[content_hash].append((memory_id, memory))
            
            # Delete duplicates, keeping newest
            for content_hash, memories in content_groups.items():
                if len(memories) > 1:
                    # Sort by timestamp, keep newest
                    memories.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                    
                    # Delete all but the newest
                    for memory_id, memory in memories[1:]:
                        if task.rule.parameters.get('backup_before_delete', False):
                            # Create backup (simplified)
                            backup_id = f"backup_{memory_id}"
                            memory['backed_up'] = True
                        
                        del self.memory_store[memory_id]
                        deleted_count += 1
            
            self.metrics.memories_deleted += deleted_count
            
            return {
                'action': 'delete',
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting memories: {e}")
            raise
    
    async def _compress_memories(self, task: ConsolidationTask) -> Dict[str, Any]:
        """Compress memories according to task"""
        try:
            compressed_count = 0
            space_saved = 0
            
            for memory_id in task.target_memories:
                if memory_id in self.memory_store:
                    memory = self.memory_store[memory_id]
                    
                    # Simple compression simulation
                    original_size = len(json.dumps(memory['content']))
                    compressed_content = self._compress_content(memory['content'])
                    compressed_size = len(json.dumps(compressed_content))
                    
                    if compressed_size < original_size:
                        memory['content'] = compressed_content
                        memory['compressed'] = True
                        memory['compression_ratio'] = compressed_size / original_size
                        
                        space_saved += original_size - compressed_size
                        compressed_count += 1
            
            self.metrics.space_saved += space_saved
            
            return {
                'action': 'compress',
                'compressed_count': compressed_count,
                'space_saved': space_saved
            }
            
        except Exception as e:
            logger.error(f"Error compressing memories: {e}")
            raise
    
    def _merge_memory_contents(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple memory contents"""
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for content in contents:
            all_keys.update(content.keys())
        
        # Merge values for each key
        for key in all_keys:
            values = [content.get(key) for content in contents if key in content]
            
            if not values:
                continue
            
            # For numeric values, use average
            if all(isinstance(v, (int, float)) for v in values):
                merged[key] = sum(values) / len(values)
            # For strings, use most common
            else:
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[str(v)] += 1
                merged[key] = max(value_counts.items(), key=lambda x: x[1])[0]
        
        return merged
    
    def _merge_memory_metadata(self, metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple memory metadata"""
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for metadata in metadatas:
            all_keys.update(metadata.keys())
        
        # Merge metadata
        for key in all_keys:
            values = [metadata.get(key) for metadata in metadatas if key in metadata]
            if values:
                # Keep first non-None value for most metadata
                merged[key] = next((v for v in values if v is not None), None)
        
        # Add merge information
        merged['merged_from'] = len(metadatas)
        merged['merge_timestamp'] = datetime.now().isoformat()
        
        return merged
    
    def _compress_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Simple content compression"""
        compressed = {}
        
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 100:
                # Simple string compression simulation
                compressed[key] = value[:50] + "..." + value[-47:]
            else:
                compressed[key] = value
        
        return compressed
    
    def _estimate_space_saved(self, memories: List[Dict[str, Any]]) -> int:
        """Estimate space saved by consolidation"""
        total_size = 0
        for memory in memories:
            total_size += len(json.dumps(memory['content']))
        
        # Assume 30% space savings from consolidation
        return int(total_size * 0.3)
    
    async def _periodic_consolidation(self):
        """Periodic consolidation task"""
        logger.info("Periodic consolidation task started")
        
        while self.running:
            try:
                await asyncio.sleep(self.consolidation_interval)
                
                if not self.running:
                    break
                
                logger.info("Running periodic consolidation...")
                
                # Check all clusters for consolidation opportunities
                for cluster in self.clusters.values():
                    if cluster.should_merge():
                        await self._schedule_consolidation_task(cluster)
                
                # Check for old memories to archive
                await self._check_for_archival()
                
                logger.info("Periodic consolidation completed")
                
            except Exception as e:
                logger.error(f"Error in periodic consolidation: {e}")
        
        logger.info("Periodic consolidation task stopped")
    
    async def _check_for_archival(self):
        """Check for memories that should be archived"""
        try:
            current_time = datetime.now()
            archive_candidates = []
            
            for memory_id, memory in self.memory_store.items():
                if memory.get('archived', False):
                    continue
                
                age = current_time - memory['timestamp']
                if age.days >= self.archive_age_days and memory['access_count'] == 0:
                    archive_candidates.append(memory_id)
            
            if archive_candidates:
                # Create archive task
                archive_rule = self.consolidation_rules.get('archive_old')
                if archive_rule:
                    task = ConsolidationTask(
                        task_id=f"archive_task_{int(time.time())}",
                        rule=archive_rule,
                        target_memories=archive_candidates,
                        scheduled_time=current_time
                    )
                    await self.task_queue.put(task)
                    
                    logger.info(f"Scheduled archival of {len(archive_candidates)} old memories")
            
        except Exception as e:
            logger.error(f"Error checking for archival: {e}")
    
    def get_metrics(self) -> ConsolidationMetrics:
        """Get current consolidation metrics"""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'memory_count': len(self.memory_store),
            'cluster_count': len(self.clusters),
            'pending_tasks': self.task_queue.qsize(),
            'worker_count': len(self.worker_tasks),
            'consolidation_rules': len(self.consolidation_rules),
            'metrics': {
                'total_tasks': self.metrics.total_tasks,
                'completed_tasks': self.metrics.completed_tasks,
                'failed_tasks': self.metrics.failed_tasks,
                'memories_processed': self.metrics.memories_processed,
                'memories_merged': self.metrics.memories_merged,
                'memories_archived': self.metrics.memories_archived,
                'memories_deleted': self.metrics.memories_deleted,
                'space_saved': self.metrics.space_saved
            }
        }

# Test and demonstration
async def test_memory_consolidator():
    """Test the memory consolidator system"""
    print("🧠 Testing CortexOS Memory Consolidator...")
    
    # Initialize consolidator
    config = {
        'worker_count': 2,
        'consolidation_interval': 10,  # 10 seconds for testing
        'similarity_threshold': 0.7,
        'archive_age_days': 1
    }
    
    consolidator = MemoryConsolidator(config)
    await consolidator.start()
    
    try:
        # Add test memories
        test_memories = [
            {
                'id': 'mem_001',
                'content': {'type': 'concept', 'name': 'neural_network', 'complexity': 'high'},
                'metadata': {'domain': 'ai', 'importance': 'critical'}
            },
            {
                'id': 'mem_002',
                'content': {'type': 'concept', 'name': 'neural_network', 'complexity': 'high'},
                'metadata': {'domain': 'ai', 'importance': 'critical'}
            },
            {
                'id': 'mem_003',
                'content': {'type': 'concept', 'name': 'machine_learning', 'complexity': 'medium'},
                'metadata': {'domain': 'ai', 'importance': 'high'}
            },
            {
                'id': 'mem_004',
                'content': {'type': 'data', 'name': 'training_set', 'size': 'large'},
                'metadata': {'domain': 'data', 'importance': 'medium'}
            }
        ]
        
        print("📝 Adding test memories...")
        for memory in test_memories:
            success = await consolidator.add_memory(
                memory['id'], memory['content'], memory['metadata']
            )
            print(f"   Added {memory['id']}: {'✅' if success else '❌'}")
        
        # Wait for consolidation to process
        print("\n⏳ Waiting for consolidation processing...")
        await asyncio.sleep(5)
        
        # Check clusters
        print(f"\n🔗 Memory Clusters: {len(consolidator.clusters)}")
        for cluster_id, cluster in consolidator.clusters.items():
            print(f"   {cluster_id}: {len(cluster.memories)} memories")
            print(f"      Similarity: {cluster._calculate_internal_similarity():.3f}")
            print(f"      Should merge: {cluster.should_merge()}")
        
        # Wait for potential consolidation tasks
        print("\n⏳ Waiting for consolidation tasks...")
        await asyncio.sleep(3)
        
        # Display metrics
        print("\n📊 Consolidation Metrics:")
        metrics = consolidator.get_metrics()
        print(f"   Total tasks: {metrics.total_tasks}")
        print(f"   Completed tasks: {metrics.completed_tasks}")
        print(f"   Failed tasks: {metrics.failed_tasks}")
        print(f"   Memories processed: {metrics.memories_processed}")
        print(f"   Memories merged: {metrics.memories_merged}")
        print(f"   Memories archived: {metrics.memories_archived}")
        print(f"   Memories deleted: {metrics.memories_deleted}")
        print(f"   Space saved: {metrics.space_saved} bytes")
        
        # Display status
        print("\n🔧 System Status:")
        status = consolidator.get_status()
        for key, value in status.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
        print("\n✅ Memory Consolidator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await consolidator.stop()

if __name__ == "__main__":
    asyncio.run(test_memory_consolidator())

