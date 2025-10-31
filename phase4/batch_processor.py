#!/usr/bin/env python3
"""
CortexOS Phase 4: Batch Processor
High-performance batch data processing system
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum
import concurrent.futures
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchType(Enum):
    """Types of batch processing"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    PIPELINE = "pipeline"

class BatchStatus(Enum):
    """Batch processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchConfig:
    """Batch processing configuration"""
    batch_id: str
    name: str
    batch_type: BatchType
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_count: int = 3
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    memory_limit_mb: int = 1024
    enable_compression: bool = False

@dataclass
class BatchJob:
    """Individual batch job"""
    job_id: str
    batch_id: str
    data: List[Any]
    processor_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_time: datetime = field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    progress: float = 0.0

@dataclass
class BatchMetrics:
    """Batch processing metrics"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0

class BatchCheckpoint:
    """Batch processing checkpoint"""
    
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.checkpoints = {}
        self.last_checkpoint_time = datetime.now()
    
    def save_checkpoint(self, job_id: str, progress: float, partial_result: Any = None):
        """Save job checkpoint"""
        self.checkpoints[job_id] = {
            'progress': progress,
            'partial_result': partial_result,
            'timestamp': datetime.now()
        }
        self.last_checkpoint_time = datetime.now()
    
    def load_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job checkpoint"""
        return self.checkpoints.get(job_id)
    
    def clear_checkpoint(self, job_id: str):
        """Clear job checkpoint"""
        if job_id in self.checkpoints:
            del self.checkpoints[job_id]
    
    def get_all_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get all checkpoints"""
        return self.checkpoints.copy()

class BatchWorker:
    """Individual batch worker"""
    
    def __init__(self, worker_id: str, processor_functions: Dict[str, Callable]):
        self.worker_id = worker_id
        self.processor_functions = processor_functions
        self.current_job = None
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = None
    
    async def process_job(self, job: BatchJob, checkpoint: BatchCheckpoint) -> BatchJob:
        """Process batch job"""
        try:
            self.current_job = job
            self.start_time = time.time()
            
            job.status = BatchStatus.RUNNING
            job.started_time = datetime.now()
            
            # Get processor function
            if job.processor_function not in self.processor_functions:
                raise ValueError(f"Unknown processor function: {job.processor_function}")
            
            processor_func = self.processor_functions[job.processor_function]
            
            # Check for existing checkpoint
            checkpoint_data = checkpoint.load_checkpoint(job.job_id)
            start_index = 0
            partial_results = []
            
            if checkpoint_data:
                start_index = int(checkpoint_data['progress'] * len(job.data))
                partial_results = checkpoint_data.get('partial_result', [])
                logger.info(f"Resuming job {job.job_id} from checkpoint at {start_index}")
            
            # Process data items
            results = partial_results.copy()
            total_items = len(job.data)
            
            for i in range(start_index, total_items):
                try:
                    # Process single item
                    item = job.data[i]
                    result = await self._process_item(processor_func, item, job.parameters)
                    results.append(result)
                    
                    self.processed_items += 1
                    job.progress = (i + 1) / total_items
                    
                    # Save checkpoint periodically
                    if (i + 1) % 100 == 0:  # Every 100 items
                        checkpoint.save_checkpoint(job.job_id, job.progress, results)
                    
                    # Allow other tasks to run
                    if i % 10 == 0:
                        await asyncio.sleep(0)
                    
                except Exception as e:
                    logger.error(f"Error processing item {i} in job {job.job_id}: {e}")
                    self.failed_items += 1
                    results.append(None)  # Placeholder for failed item
            
            # Complete job
            job.status = BatchStatus.COMPLETED
            job.completed_time = datetime.now()
            job.result = results
            job.progress = 1.0
            
            # Clear checkpoint
            checkpoint.clear_checkpoint(job.job_id)
            
            logger.info(f"Worker {self.worker_id} completed job {job.job_id}")
            return job
            
        except Exception as e:
            job.status = BatchStatus.FAILED
            job.error_message = str(e)
            job.completed_time = datetime.now()
            logger.error(f"Worker {self.worker_id} failed job {job.job_id}: {e}")
            return job
        finally:
            self.current_job = None
    
    async def _process_item(self, processor_func: Callable, item: Any, parameters: Dict[str, Any]) -> Any:
        """Process single data item"""
        try:
            # Call processor function
            if asyncio.iscoroutinefunction(processor_func):
                result = await processor_func(item, **parameters)
            else:
                result = processor_func(item, **parameters)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status"""
        return {
            'worker_id': self.worker_id,
            'current_job': self.current_job.job_id if self.current_job else None,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }

class BatchProcessor:
    """High-performance batch data processor"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.job_queue = asyncio.Queue()
        self.completed_jobs = {}
        self.failed_jobs = {}
        self.workers = []
        self.checkpoint = BatchCheckpoint(config.batch_id)
        self.metrics = BatchMetrics()
        self.processor_functions = {}
        self.running = False
        self.worker_tasks = []
        
        logger.info(f"Batch Processor initialized: {config.batch_id}")
    
    def register_processor(self, name: str, function: Callable):
        """Register processor function"""
        self.processor_functions[name] = function
        logger.info(f"Registered processor function: {name}")
    
    async def start(self):
        """Start batch processing"""
        try:
            self.running = True
            
            # Create workers
            for i in range(self.config.max_workers):
                worker = BatchWorker(f"worker_{i}", self.processor_functions)
                self.workers.append(worker)
            
            # Start worker tasks
            for worker in self.workers:
                task = asyncio.create_task(self._worker_loop(worker))
                self.worker_tasks.append(task)
            
            logger.info(f"Batch Processor started with {self.config.max_workers} workers")
            
        except Exception as e:
            logger.error(f"Error starting Batch Processor: {e}")
            raise
    
    async def stop(self):
        """Stop batch processing"""
        try:
            self.running = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            logger.info("Batch Processor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Batch Processor: {e}")
    
    async def submit_job(self, job: BatchJob) -> str:
        """Submit batch job for processing"""
        try:
            # Validate job
            if job.processor_function not in self.processor_functions:
                raise ValueError(f"Unknown processor function: {job.processor_function}")
            
            # Add to queue
            await self.job_queue.put(job)
            self.metrics.total_jobs += 1
            self.metrics.total_items += len(job.data)
            
            logger.info(f"Submitted batch job {job.job_id} with {len(job.data)} items")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get job status"""
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            return self.failed_jobs[job_id]
        
        # Check running jobs
        for worker in self.workers:
            if worker.current_job and worker.current_job.job_id == job_id:
                return worker.current_job
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel running job"""
        try:
            # Find and cancel job
            for worker in self.workers:
                if worker.current_job and worker.current_job.job_id == job_id:
                    worker.current_job.status = BatchStatus.CANCELLED
                    self.metrics.cancelled_jobs += 1
                    logger.info(f"Cancelled job {job_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    async def _worker_loop(self, worker: BatchWorker):
        """Main worker loop"""
        logger.info(f"Worker {worker.worker_id} started")
        
        while self.running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Process job
                completed_job = await worker.process_job(job, self.checkpoint)
                
                # Store result
                if completed_job.status == BatchStatus.COMPLETED:
                    self.completed_jobs[completed_job.job_id] = completed_job
                    self.metrics.completed_jobs += 1
                    self.metrics.processed_items += len(completed_job.data)
                elif completed_job.status == BatchStatus.FAILED:
                    self.failed_jobs[completed_job.job_id] = completed_job
                    self.metrics.failed_jobs += 1
                    self.metrics.failed_items += len(completed_job.data)
                
                # Update metrics
                if completed_job.started_time and completed_job.completed_time:
                    processing_time = (completed_job.completed_time - completed_job.started_time).total_seconds()
                    self.metrics.average_processing_time = (
                        (self.metrics.average_processing_time * (self.metrics.completed_jobs - 1) + processing_time)
                        / self.metrics.completed_jobs
                    ) if self.metrics.completed_jobs > 0 else processing_time
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in worker {worker.worker_id}: {e}")
        
        logger.info(f"Worker {worker.worker_id} stopped")
    
    def get_metrics(self) -> BatchMetrics:
        """Get current batch metrics"""
        # Calculate throughput
        if self.metrics.average_processing_time > 0:
            self.metrics.throughput_per_second = 1.0 / self.metrics.average_processing_time
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            'batch_id': self.config.batch_id,
            'running': self.running,
            'pending_jobs': self.job_queue.qsize(),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'workers': [worker.get_status() for worker in self.workers],
            'registered_processors': list(self.processor_functions.keys()),
            'metrics': {
                'total_jobs': self.metrics.total_jobs,
                'completed_jobs': self.metrics.completed_jobs,
                'failed_jobs': self.metrics.failed_jobs,
                'cancelled_jobs': self.metrics.cancelled_jobs,
                'total_items': self.metrics.total_items,
                'processed_items': self.metrics.processed_items,
                'failed_items': self.metrics.failed_items,
                'average_processing_time': self.metrics.average_processing_time,
                'throughput_per_second': self.metrics.throughput_per_second
            }
        }

class BatchProcessorManager:
    """Manager for multiple batch processors"""
    
    def __init__(self):
        self.processors = {}
        self.running = False
    
    async def start(self):
        """Start all processors"""
        self.running = True
        for processor in self.processors.values():
            await processor.start()
        logger.info("Batch Processor Manager started")
    
    async def stop(self):
        """Stop all processors"""
        self.running = False
        for processor in self.processors.values():
            await processor.stop()
        logger.info("Batch Processor Manager stopped")
    
    def create_processor(self, config: BatchConfig) -> BatchProcessor:
        """Create new batch processor"""
        processor = BatchProcessor(config)
        self.processors[config.batch_id] = processor
        logger.info(f"Created batch processor: {config.batch_id}")
        return processor
    
    def get_processor(self, batch_id: str) -> Optional[BatchProcessor]:
        """Get processor by batch ID"""
        return self.processors.get(batch_id)
    
    def remove_processor(self, batch_id: str) -> bool:
        """Remove processor"""
        if batch_id in self.processors:
            del self.processors[batch_id]
            logger.info(f"Removed batch processor: {batch_id}")
            return True
        return False
    
    def get_all_metrics(self) -> Dict[str, BatchMetrics]:
        """Get metrics for all processors"""
        return {batch_id: processor.get_metrics() 
                for batch_id, processor in self.processors.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_jobs = sum(p.metrics.total_jobs for p in self.processors.values())
        total_completed = sum(p.metrics.completed_jobs for p in self.processors.values())
        total_failed = sum(p.metrics.failed_jobs for p in self.processors.values())
        total_items = sum(p.metrics.total_items for p in self.processors.values())
        
        return {
            'running': self.running,
            'processor_count': len(self.processors),
            'total_jobs': total_jobs,
            'total_completed_jobs': total_completed,
            'total_failed_jobs': total_failed,
            'total_items': total_items,
            'processors': {batch_id: processor.get_status() 
                          for batch_id, processor in self.processors.items()}
        }

# Example processor functions
async def neural_data_processor(item: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Example neural data processor"""
    # Simulate neural data processing
    await asyncio.sleep(0.01)  # Simulate processing time
    
    processed_item = item.copy()
    processed_item['processed'] = True
    processed_item['processing_time'] = datetime.now().isoformat()
    
    # Add some processing results
    if 'signal' in item:
        processed_item['normalized_signal'] = item['signal'] * kwargs.get('normalization_factor', 1.0)
    
    return processed_item

def text_data_processor(item: str, **kwargs) -> Dict[str, Any]:
    """Example text data processor"""
    # Simulate text processing
    return {
        'original_text': item,
        'length': len(item),
        'word_count': len(item.split()),
        'uppercase': item.upper(),
        'processed': True
    }

def numeric_data_processor(item: float, **kwargs) -> Dict[str, Any]:
    """Example numeric data processor"""
    multiplier = kwargs.get('multiplier', 1.0)
    offset = kwargs.get('offset', 0.0)
    
    return {
        'original_value': item,
        'processed_value': item * multiplier + offset,
        'squared': item ** 2,
        'processed': True
    }

# Test and demonstration
async def test_batch_processor():
    """Test the batch processor system"""
    print("🧠 Testing CortexOS Batch Processor...")
    
    # Create batch configurations
    neural_config = BatchConfig(
        batch_id="neural_batch",
        name="Neural Data Batch",
        batch_type=BatchType.PARALLEL,
        batch_size=100,
        max_workers=3,
        timeout_seconds=60
    )
    
    text_config = BatchConfig(
        batch_id="text_batch",
        name="Text Data Batch",
        batch_type=BatchType.SEQUENTIAL,
        batch_size=50,
        max_workers=2,
        timeout_seconds=30
    )
    
    # Create processor manager
    manager = BatchProcessorManager()
    
    # Create processors
    neural_processor = manager.create_processor(neural_config)
    text_processor = manager.create_processor(text_config)
    
    # Register processor functions
    neural_processor.register_processor("neural_processor", neural_data_processor)
    text_processor.register_processor("text_processor", text_data_processor)
    
    try:
        # Start processors
        await manager.start()
        print("✅ Batch processors started")
        
        # Create test data
        neural_data = [
            {'signal': i * 0.1, 'timestamp': datetime.now().isoformat()}
            for i in range(50)
        ]
        
        text_data = [
            f"This is test sentence number {i} for processing."
            for i in range(25)
        ]
        
        # Submit jobs
        print("\n📤 Submitting batch jobs...")
        
        neural_job = BatchJob(
            job_id="neural_job_001",
            batch_id="neural_batch",
            data=neural_data,
            processor_function="neural_processor",
            parameters={'normalization_factor': 2.0}
        )
        
        text_job = BatchJob(
            job_id="text_job_001",
            batch_id="text_batch",
            data=text_data,
            processor_function="text_processor"
        )
        
        neural_job_id = await neural_processor.submit_job(neural_job)
        text_job_id = await text_processor.submit_job(text_job)
        
        print(f"   Neural job: {neural_job_id}")
        print(f"   Text job: {text_job_id}")
        
        # Monitor progress
        print("\n⏳ Processing jobs...")
        
        while True:
            neural_status = await neural_processor.get_job_status(neural_job_id)
            text_status = await text_processor.get_job_status(text_job_id)
            
            print(f"\r   Neural: {neural_status.progress:.1%} | Text: {text_status.progress:.1%}", end="")
            
            if (neural_status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED] and
                text_status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]):
                break
            
            await asyncio.sleep(0.5)
        
        print("\n")
        
        # Display results
        print("📊 Job Results:")
        neural_final = await neural_processor.get_job_status(neural_job_id)
        text_final = await text_processor.get_job_status(text_job_id)
        
        print(f"   Neural job: {neural_final.status.value}")
        if neural_final.result:
            print(f"      Processed {len(neural_final.result)} items")
        
        print(f"   Text job: {text_final.status.value}")
        if text_final.result:
            print(f"      Processed {len(text_final.result)} items")
        
        # Display metrics
        print("\n📈 Batch Metrics:")
        all_metrics = manager.get_all_metrics()
        for batch_id, metrics in all_metrics.items():
            print(f"   {batch_id}:")
            print(f"      Total jobs: {metrics.total_jobs}")
            print(f"      Completed: {metrics.completed_jobs}")
            print(f"      Failed: {metrics.failed_jobs}")
            print(f"      Items processed: {metrics.processed_items}")
            print(f"      Avg processing time: {metrics.average_processing_time:.3f}s")
            print(f"      Throughput: {metrics.throughput_per_second:.2f} jobs/sec")
        
        # Display system status
        print("\n🔧 System Status:")
        status = manager.get_system_status()
        print(f"   Running: {status['running']}")
        print(f"   Processors: {status['processor_count']}")
        print(f"   Total jobs: {status['total_jobs']}")
        print(f"   Completed jobs: {status['total_completed_jobs']}")
        print(f"   Failed jobs: {status['total_failed_jobs']}")
        
        print("\n✅ Batch Processor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(test_batch_processor())

