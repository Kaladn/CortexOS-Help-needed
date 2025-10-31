#!/usr/bin/env python3
"""
CortexOS Phase 3: Memory Retriever
Advanced neural memory retrieval and pattern matching system
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
class MemoryQuery:
    """Memory retrieval query structure"""
    query_id: str
    pattern: Dict[str, Any]
    similarity_threshold: float = 0.7
    max_results: int = 100
    time_range: Optional[Tuple[datetime, datetime]] = None
    priority: int = 1
    context_filters: Dict[str, Any] = field(default_factory=dict)
    semantic_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class MemoryMatch:
    """Memory retrieval match result"""
    memory_id: str
    content: Dict[str, Any]
    similarity_score: float
    confidence: float
    retrieval_time: datetime
    context_match: bool
    semantic_relevance: float
    access_count: int = 0

@dataclass
class RetrievalMetrics:
    """Memory retrieval performance metrics"""
    total_queries: int = 0
    successful_retrievals: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    pattern_match_accuracy: float = 0.0
    semantic_relevance_score: float = 0.0

class MemoryIndex:
    """Advanced memory indexing system"""
    
    def __init__(self):
        self.pattern_index = defaultdict(set)
        self.semantic_index = defaultdict(list)
        self.temporal_index = defaultdict(list)
        self.context_index = defaultdict(set)
        self.similarity_cache = {}
        self.access_patterns = defaultdict(int)
        
    def add_memory(self, memory_id: str, content: Dict[str, Any], 
                   timestamp: datetime, context: Dict[str, Any]):
        """Add memory to all relevant indices"""
        try:
            # Pattern indexing
            for key, value in content.items():
                pattern_key = f"{key}:{str(value)[:50]}"
                self.pattern_index[pattern_key].add(memory_id)
            
            # Semantic indexing (simplified)
            semantic_signature = self._generate_semantic_signature(content)
            self.semantic_index[semantic_signature].append(memory_id)
            
            # Temporal indexing
            time_bucket = timestamp.replace(minute=0, second=0, microsecond=0)
            self.temporal_index[time_bucket].append(memory_id)
            
            # Context indexing
            for ctx_key, ctx_value in context.items():
                ctx_pattern = f"{ctx_key}:{str(ctx_value)}"
                self.context_index[ctx_pattern].add(memory_id)
                
            logger.debug(f"Indexed memory {memory_id} across all indices")
            
        except Exception as e:
            logger.error(f"Error indexing memory {memory_id}: {e}")
    
    def _generate_semantic_signature(self, content: Dict[str, Any]) -> str:
        """Generate semantic signature for content"""
        try:
            # Simple semantic signature based on content structure and types
            signature_parts = []
            
            for key, value in sorted(content.items()):
                value_type = type(value).__name__
                value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
                signature_parts.append(f"{key}:{value_type}:{value_hash}")
            
            return hashlib.md5("|".join(signature_parts).encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Error generating semantic signature: {e}")
            return "unknown"
    
    def find_candidates(self, query: MemoryQuery) -> Set[str]:
        """Find candidate memory IDs for retrieval"""
        candidates = set()
        
        try:
            # Pattern-based candidates
            for key, value in query.pattern.items():
                pattern_key = f"{key}:{str(value)[:50]}"
                if pattern_key in self.pattern_index:
                    candidates.update(self.pattern_index[pattern_key])
            
            # Context-based candidates
            for ctx_key, ctx_value in query.context_filters.items():
                ctx_pattern = f"{ctx_key}:{str(ctx_value)}"
                if ctx_pattern in self.context_index:
                    candidates.update(self.context_index[ctx_pattern])
            
            # Temporal candidates
            if query.time_range:
                start_time, end_time = query.time_range
                current_time = start_time.replace(minute=0, second=0, microsecond=0)
                
                while current_time <= end_time:
                    if current_time in self.temporal_index:
                        candidates.update(self.temporal_index[current_time])
                    current_time += timedelta(hours=1)
            
            logger.debug(f"Found {len(candidates)} candidates for query {query.query_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding candidates: {e}")
            return set()

class MemoryRetriever:
    """Advanced neural memory retrieval system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_store = {}
        self.memory_index = MemoryIndex()
        self.retrieval_cache = {}
        self.metrics = RetrievalMetrics()
        self.active_queries = {}
        self.query_queue = asyncio.Queue()
        self.running = False
        self.worker_tasks = []
        
        # Configuration
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.worker_count = self.config.get('worker_count', 4)
        self.similarity_algorithm = self.config.get('similarity_algorithm', 'cosine')
        
        logger.info("Memory Retriever initialized")
    
    async def start(self):
        """Start the memory retrieval system"""
        try:
            self.running = True
            
            # Start worker tasks
            for i in range(self.worker_count):
                task = asyncio.create_task(self._query_worker(f"worker_{i}"))
                self.worker_tasks.append(task)
            
            logger.info(f"Memory Retriever started with {self.worker_count} workers")
            
        except Exception as e:
            logger.error(f"Error starting Memory Retriever: {e}")
            raise
    
    async def stop(self):
        """Stop the memory retrieval system"""
        try:
            self.running = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            logger.info("Memory Retriever stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Memory Retriever: {e}")
    
    async def store_memory(self, memory_id: str, content: Dict[str, Any], 
                          context: Dict[str, Any] = None) -> bool:
        """Store memory for future retrieval"""
        try:
            timestamp = datetime.now()
            context = context or {}
            
            # Store memory
            self.memory_store[memory_id] = {
                'content': content,
                'context': context,
                'timestamp': timestamp,
                'access_count': 0,
                'last_accessed': timestamp
            }
            
            # Index memory
            self.memory_index.add_memory(memory_id, content, timestamp, context)
            
            logger.debug(f"Stored memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory {memory_id}: {e}")
            return False
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Retrieve memories matching the query"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.retrieval_cache:
                cache_entry = self.retrieval_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.metrics.cache_hit_rate = (self.metrics.cache_hit_rate * self.metrics.total_queries + 1) / (self.metrics.total_queries + 1)
                    logger.debug(f"Cache hit for query {query.query_id}")
                    return cache_entry['results']
            
            # Add query to processing queue
            result_future = asyncio.Future()
            await self.query_queue.put((query, result_future))
            
            # Wait for results
            results = await result_future
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.total_queries += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_queries - 1) + processing_time) 
                / self.metrics.total_queries
            )
            
            if results:
                self.metrics.successful_retrievals += 1
            
            # Cache results
            self._cache_results(cache_key, results)
            
            logger.info(f"Retrieved {len(results)} memories for query {query.query_id} in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories for query {query.query_id}: {e}")
            return []
    
    async def _query_worker(self, worker_id: str):
        """Worker task for processing retrieval queries"""
        logger.info(f"Query worker {worker_id} started")
        
        while self.running:
            try:
                # Get query from queue
                query, result_future = await asyncio.wait_for(
                    self.query_queue.get(), timeout=1.0
                )
                
                # Process query
                results = await self._process_query(query)
                
                # Return results
                if not result_future.cancelled():
                    result_future.set_result(results)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in query worker {worker_id}: {e}")
                if not result_future.cancelled():
                    result_future.set_result([])
        
        logger.info(f"Query worker {worker_id} stopped")
    
    async def _process_query(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Process a single retrieval query"""
        try:
            # Find candidate memories
            candidates = self.memory_index.find_candidates(query)
            
            if not candidates:
                logger.debug(f"No candidates found for query {query.query_id}")
                return []
            
            # Score and rank candidates
            matches = []
            for memory_id in candidates:
                if memory_id not in self.memory_store:
                    continue
                
                memory_data = self.memory_store[memory_id]
                similarity_score = self._calculate_similarity(
                    query.pattern, memory_data['content']
                )
                
                if similarity_score >= query.similarity_threshold:
                    # Calculate additional metrics
                    confidence = self._calculate_confidence(
                        similarity_score, memory_data['access_count']
                    )
                    
                    context_match = self._check_context_match(
                        query.context_filters, memory_data['context']
                    )
                    
                    semantic_relevance = self._calculate_semantic_relevance(
                        query, memory_data['content']
                    )
                    
                    match = MemoryMatch(
                        memory_id=memory_id,
                        content=memory_data['content'],
                        similarity_score=similarity_score,
                        confidence=confidence,
                        retrieval_time=datetime.now(),
                        context_match=context_match,
                        semantic_relevance=semantic_relevance,
                        access_count=memory_data['access_count']
                    )
                    
                    matches.append(match)
                    
                    # Update access statistics
                    memory_data['access_count'] += 1
                    memory_data['last_accessed'] = datetime.now()
            
            # Sort by combined score
            matches.sort(key=lambda m: (
                m.similarity_score * 0.4 + 
                m.confidence * 0.3 + 
                m.semantic_relevance * 0.3
            ), reverse=True)
            
            # Limit results
            matches = matches[:query.max_results]
            
            logger.debug(f"Processed query {query.query_id}: {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error processing query {query.query_id}: {e}")
            return []
    
    def _calculate_similarity(self, pattern: Dict[str, Any], content: Dict[str, Any]) -> float:
        """Calculate similarity between pattern and content"""
        try:
            if self.similarity_algorithm == 'cosine':
                return self._cosine_similarity(pattern, content)
            elif self.similarity_algorithm == 'jaccard':
                return self._jaccard_similarity(pattern, content)
            else:
                return self._simple_similarity(pattern, content)
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, pattern: Dict[str, Any], content: Dict[str, Any]) -> float:
        """Calculate cosine similarity between pattern and content"""
        try:
            # Convert to feature vectors
            all_keys = set(pattern.keys()) | set(content.keys())
            
            pattern_vector = []
            content_vector = []
            
            for key in all_keys:
                pattern_val = 1.0 if key in pattern else 0.0
                content_val = 1.0 if key in content else 0.0
                
                pattern_vector.append(pattern_val)
                content_vector.append(content_val)
            
            # Calculate cosine similarity
            pattern_array = np.array(pattern_vector)
            content_array = np.array(content_vector)
            
            dot_product = np.dot(pattern_array, content_array)
            norm_pattern = np.linalg.norm(pattern_array)
            norm_content = np.linalg.norm(content_array)
            
            if norm_pattern == 0 or norm_content == 0:
                return 0.0
            
            return dot_product / (norm_pattern * norm_content)
            
        except Exception as e:
            logger.error(f"Error in cosine similarity: {e}")
            return 0.0
    
    def _jaccard_similarity(self, pattern: Dict[str, Any], content: Dict[str, Any]) -> float:
        """Calculate Jaccard similarity between pattern and content"""
        try:
            pattern_keys = set(pattern.keys())
            content_keys = set(content.keys())
            
            intersection = len(pattern_keys & content_keys)
            union = len(pattern_keys | content_keys)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in Jaccard similarity: {e}")
            return 0.0
    
    def _simple_similarity(self, pattern: Dict[str, Any], content: Dict[str, Any]) -> float:
        """Calculate simple similarity between pattern and content"""
        try:
            matches = 0
            total = len(pattern)
            
            for key, value in pattern.items():
                if key in content and str(content[key]) == str(value):
                    matches += 1
            
            return matches / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in simple similarity: {e}")
            return 0.0
    
    def _calculate_confidence(self, similarity_score: float, access_count: int) -> float:
        """Calculate confidence score based on similarity and access patterns"""
        try:
            # Base confidence from similarity
            base_confidence = similarity_score
            
            # Boost from access patterns (frequently accessed memories are more reliable)
            access_boost = min(access_count * 0.01, 0.2)  # Max 20% boost
            
            return min(base_confidence + access_boost, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return similarity_score
    
    def _check_context_match(self, query_context: Dict[str, Any], 
                           memory_context: Dict[str, Any]) -> bool:
        """Check if memory context matches query context filters"""
        try:
            if not query_context:
                return True
            
            for key, value in query_context.items():
                if key not in memory_context or memory_context[key] != value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking context match: {e}")
            return False
    
    def _calculate_semantic_relevance(self, query: MemoryQuery, 
                                    content: Dict[str, Any]) -> float:
        """Calculate semantic relevance score"""
        try:
            if not query.semantic_weights:
                return 0.5  # Default relevance
            
            relevance_score = 0.0
            total_weight = 0.0
            
            for key, weight in query.semantic_weights.items():
                if key in content:
                    relevance_score += weight
                total_weight += weight
            
            return relevance_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic relevance: {e}")
            return 0.0
    
    def _generate_cache_key(self, query: MemoryQuery) -> str:
        """Generate cache key for query"""
        try:
            key_data = {
                'pattern': query.pattern,
                'threshold': query.similarity_threshold,
                'max_results': query.max_results,
                'context_filters': query.context_filters,
                'semantic_weights': query.semantic_weights
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"query_{query.query_id}"
    
    def _cache_results(self, cache_key: str, results: List[MemoryMatch]):
        """Cache retrieval results"""
        try:
            # Clean old cache entries if needed
            if len(self.retrieval_cache) >= self.max_cache_size:
                self._clean_cache()
            
            # Store results
            self.retrieval_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def _clean_cache(self):
        """Clean old cache entries"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.retrieval_cache.items():
                if current_time - entry['timestamp'] > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.retrieval_cache[key]
            
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def get_metrics(self) -> RetrievalMetrics:
        """Get current retrieval metrics"""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'memory_count': len(self.memory_store),
            'cache_size': len(self.retrieval_cache),
            'active_queries': len(self.active_queries),
            'worker_count': len(self.worker_tasks),
            'metrics': {
                'total_queries': self.metrics.total_queries,
                'successful_retrievals': self.metrics.successful_retrievals,
                'average_response_time': self.metrics.average_response_time,
                'cache_hit_rate': self.metrics.cache_hit_rate
            }
        }

# Test and demonstration
async def test_memory_retriever():
    """Test the memory retriever system"""
    print("🧠 Testing CortexOS Memory Retriever...")
    
    # Initialize retriever
    config = {
        'max_cache_size': 1000,
        'cache_ttl': 300,
        'worker_count': 2,
        'similarity_algorithm': 'cosine'
    }
    
    retriever = MemoryRetriever(config)
    await retriever.start()
    
    try:
        # Store test memories
        test_memories = [
            {
                'id': 'mem_001',
                'content': {'type': 'concept', 'name': 'neural_network', 'complexity': 'high'},
                'context': {'domain': 'ai', 'importance': 'critical'}
            },
            {
                'id': 'mem_002', 
                'content': {'type': 'concept', 'name': 'machine_learning', 'complexity': 'medium'},
                'context': {'domain': 'ai', 'importance': 'high'}
            },
            {
                'id': 'mem_003',
                'content': {'type': 'data', 'name': 'training_set', 'size': 'large'},
                'context': {'domain': 'data', 'importance': 'medium'}
            }
        ]
        
        print("📝 Storing test memories...")
        for memory in test_memories:
            success = await retriever.store_memory(
                memory['id'], memory['content'], memory['context']
            )
            print(f"   Stored {memory['id']}: {'✅' if success else '❌'}")
        
        # Test retrieval queries
        print("\n🔍 Testing memory retrieval...")
        
        # Query 1: Find AI concepts
        query1 = MemoryQuery(
            query_id="test_query_1",
            pattern={'type': 'concept'},
            context_filters={'domain': 'ai'},
            similarity_threshold=0.5,
            max_results=10
        )
        
        results1 = await retriever.retrieve_memories(query1)
        print(f"   AI concepts query: {len(results1)} results")
        for result in results1:
            print(f"      {result.memory_id}: {result.similarity_score:.3f} similarity")
        
        # Query 2: Find high complexity items
        query2 = MemoryQuery(
            query_id="test_query_2",
            pattern={'complexity': 'high'},
            similarity_threshold=0.8,
            max_results=5
        )
        
        results2 = await retriever.retrieve_memories(query2)
        print(f"   High complexity query: {len(results2)} results")
        for result in results2:
            print(f"      {result.memory_id}: {result.similarity_score:.3f} similarity")
        
        # Test cache hit
        print("\n💾 Testing cache functionality...")
        results1_cached = await retriever.retrieve_memories(query1)
        print(f"   Cached query: {len(results1_cached)} results (should be cached)")
        
        # Display metrics
        print("\n📊 Retrieval Metrics:")
        metrics = retriever.get_metrics()
        print(f"   Total queries: {metrics.total_queries}")
        print(f"   Successful retrievals: {metrics.successful_retrievals}")
        print(f"   Average response time: {metrics.average_response_time:.3f}s")
        print(f"   Cache hit rate: {metrics.cache_hit_rate:.3f}")
        
        # Display status
        print("\n🔧 System Status:")
        status = retriever.get_status()
        for key, value in status.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
        print("\n✅ Memory Retriever test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await retriever.stop()

if __name__ == "__main__":
    asyncio.run(test_memory_retriever())

