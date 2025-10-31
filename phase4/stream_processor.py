#!/usr/bin/env python3
"""
CortexOS Phase 4: Stream Processor
Real-time neural data stream processing system
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Types of data streams"""
    NEURAL_SIGNALS = "neural_signals"
    SENSOR_DATA = "sensor_data"
    TEXT_STREAM = "text_stream"
    NUMERIC_STREAM = "numeric_stream"
    EVENT_STREAM = "event_stream"
    BINARY_STREAM = "binary_stream"

class ProcessingMode(Enum):
    """Stream processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    SLIDING_WINDOW = "sliding_window"
    TUMBLING_WINDOW = "tumbling_window"

@dataclass
class StreamConfig:
    """Stream configuration"""
    stream_id: str
    name: str
    stream_type: StreamType
    processing_mode: ProcessingMode
    buffer_size: int = 10000
    batch_size: int = 100
    window_size: int = 1000
    window_slide: int = 100
    max_latency_ms: int = 100
    enable_backpressure: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1000

@dataclass
class StreamMessage:
    """Individual stream message"""
    message_id: str
    stream_id: str
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    partition_key: Optional[str] = None

@dataclass
class ProcessingResult:
    """Stream processing result"""
    result_id: str
    stream_id: str
    processed_data: Any
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamMetrics:
    """Stream processing metrics"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0
    throughput_per_second: float = 0.0
    backpressure_events: int = 0
    checkpoint_count: int = 0
    last_checkpoint: Optional[datetime] = None

class StreamBuffer:
    """High-performance stream buffer"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.lock)
        self.not_full = asyncio.Condition(self.lock)
        self.sequence_counter = 0
        
    async def put(self, message: StreamMessage) -> bool:
        """Add message to buffer"""
        async with self.not_full:
            while len(self.buffer) >= self.max_size:
                await self.not_full.wait()
            
            message.sequence_number = self.sequence_counter
            self.sequence_counter += 1
            self.buffer.append(message)
            self.not_empty.notify()
            return True
    
    async def get(self) -> StreamMessage:
        """Get message from buffer"""
        async with self.not_empty:
            while len(self.buffer) == 0:
                await self.not_empty.wait()
            
            message = self.buffer.popleft()
            self.not_full.notify()
            return message
    
    async def get_batch(self, batch_size: int) -> List[StreamMessage]:
        """Get batch of messages from buffer"""
        batch = []
        async with self.not_empty:
            while len(batch) < batch_size and len(self.buffer) > 0:
                batch.append(self.buffer.popleft())
            
            if batch:
                self.not_full.notify()
        
        return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.max_size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0

class WindowManager:
    """Sliding and tumbling window manager"""
    
    def __init__(self, window_size: int, slide_size: int = None):
        self.window_size = window_size
        self.slide_size = slide_size or window_size
        self.windows = {}
        self.current_window = deque()
        self.window_start_time = time.time()
        
    def add_message(self, message: StreamMessage) -> List[List[StreamMessage]]:
        """Add message to window and return completed windows"""
        completed_windows = []
        current_time = time.time()
        
        # Add to current window
        self.current_window.append(message)
        
        # Check if window is complete
        if len(self.current_window) >= self.window_size:
            completed_windows.append(list(self.current_window))
            
            # Slide window
            for _ in range(self.slide_size):
                if self.current_window:
                    self.current_window.popleft()
            
            self.window_start_time = current_time
        
        return completed_windows
    
    def get_current_window(self) -> List[StreamMessage]:
        """Get current window contents"""
        return list(self.current_window)
    
    def force_window_completion(self) -> List[StreamMessage]:
        """Force completion of current window"""
        if self.current_window:
            window = list(self.current_window)
            self.current_window.clear()
            return window
        return []

class StreamProcessor:
    """Real-time neural data stream processor"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer = StreamBuffer(config.buffer_size)
        self.window_manager = WindowManager(config.window_size, config.window_slide)
        self.metrics = StreamMetrics()
        self.processors = {}
        self.running = False
        self.worker_tasks = []
        self.checkpoint_data = {}
        
        # Processing functions
        self.message_handlers = []
        self.batch_handlers = []
        self.window_handlers = []
        
        logger.info(f"Stream Processor initialized for stream {config.stream_id}")
    
    async def start(self):
        """Start stream processing"""
        try:
            self.running = True
            
            # Start processing tasks based on mode
            if self.config.processing_mode == ProcessingMode.REAL_TIME:
                task = asyncio.create_task(self._real_time_processor())
                self.worker_tasks.append(task)
            elif self.config.processing_mode == ProcessingMode.BATCH:
                task = asyncio.create_task(self._batch_processor())
                self.worker_tasks.append(task)
            elif self.config.processing_mode == ProcessingMode.MICRO_BATCH:
                task = asyncio.create_task(self._micro_batch_processor())
                self.worker_tasks.append(task)
            elif self.config.processing_mode in [ProcessingMode.SLIDING_WINDOW, ProcessingMode.TUMBLING_WINDOW]:
                task = asyncio.create_task(self._window_processor())
                self.worker_tasks.append(task)
            
            # Start checkpoint task if enabled
            if self.config.enable_checkpointing:
                checkpoint_task = asyncio.create_task(self._checkpoint_manager())
                self.worker_tasks.append(checkpoint_task)
            
            logger.info(f"Stream Processor started for {self.config.stream_id}")
            
        except Exception as e:
            logger.error(f"Error starting Stream Processor: {e}")
            raise
    
    async def stop(self):
        """Stop stream processing"""
        try:
            self.running = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            # Final checkpoint
            if self.config.enable_checkpointing:
                await self._create_checkpoint()
            
            logger.info(f"Stream Processor stopped for {self.config.stream_id}")
            
        except Exception as e:
            logger.error(f"Error stopping Stream Processor: {e}")
    
    async def send_message(self, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Send message to stream"""
        try:
            message = StreamMessage(
                message_id=f"{self.config.stream_id}_{int(time.time() * 1000000)}",
                stream_id=self.config.stream_id,
                timestamp=datetime.now(),
                data=data,
                metadata=metadata or {}
            )
            
            # Check backpressure
            if self.config.enable_backpressure and self.buffer.is_full():
                self.metrics.backpressure_events += 1
                logger.warning(f"Backpressure detected for stream {self.config.stream_id}")
                return False
            
            await self.buffer.put(message)
            self.metrics.messages_received += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def add_message_handler(self, handler: Callable[[StreamMessage], Any]):
        """Add message handler for real-time processing"""
        self.message_handlers.append(handler)
    
    def add_batch_handler(self, handler: Callable[[List[StreamMessage]], Any]):
        """Add batch handler for batch processing"""
        self.batch_handlers.append(handler)
    
    def add_window_handler(self, handler: Callable[[List[StreamMessage]], Any]):
        """Add window handler for window processing"""
        self.window_handlers.append(handler)
    
    async def _real_time_processor(self):
        """Real-time message processor"""
        logger.info(f"Real-time processor started for {self.config.stream_id}")
        
        while self.running:
            try:
                # Get message from buffer
                message = await asyncio.wait_for(self.buffer.get(), timeout=1.0)
                
                # Process message
                await self._process_single_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in real-time processor: {e}")
                self.metrics.messages_failed += 1
        
        logger.info(f"Real-time processor stopped for {self.config.stream_id}")
    
    async def _batch_processor(self):
        """Batch message processor"""
        logger.info(f"Batch processor started for {self.config.stream_id}")
        
        while self.running:
            try:
                # Get batch from buffer
                batch = await self._get_batch()
                
                if batch:
                    # Process batch
                    await self._process_message_batch(batch)
                else:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                self.metrics.messages_failed += len(batch) if 'batch' in locals() else 1
        
        logger.info(f"Batch processor stopped for {self.config.stream_id}")
    
    async def _micro_batch_processor(self):
        """Micro-batch message processor"""
        logger.info(f"Micro-batch processor started for {self.config.stream_id}")
        
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Try to get message with timeout
                try:
                    message = await asyncio.wait_for(self.buffer.get(), timeout=0.01)
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                current_time = time.time()
                
                # Process batch if size reached or timeout
                if (len(batch) >= self.config.batch_size or 
                    (batch and current_time - last_batch_time > self.config.max_latency_ms / 1000)):
                    
                    if batch:
                        await self._process_message_batch(batch)
                        batch = []
                        last_batch_time = current_time
                
                await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in micro-batch processor: {e}")
                self.metrics.messages_failed += len(batch) if batch else 1
                batch = []
        
        # Process remaining batch
        if batch:
            await self._process_message_batch(batch)
        
        logger.info(f"Micro-batch processor stopped for {self.config.stream_id}")
    
    async def _window_processor(self):
        """Window-based message processor"""
        logger.info(f"Window processor started for {self.config.stream_id}")
        
        while self.running:
            try:
                # Get message from buffer
                message = await asyncio.wait_for(self.buffer.get(), timeout=1.0)
                
                # Add to window manager
                completed_windows = self.window_manager.add_message(message)
                
                # Process completed windows
                for window in completed_windows:
                    await self._process_message_window(window)
                
            except asyncio.TimeoutError:
                # Check for forced window completion
                current_window = self.window_manager.force_window_completion()
                if current_window:
                    await self._process_message_window(current_window)
                continue
            except Exception as e:
                logger.error(f"Error in window processor: {e}")
                self.metrics.messages_failed += 1
        
        logger.info(f"Window processor stopped for {self.config.stream_id}")
    
    async def _get_batch(self) -> List[StreamMessage]:
        """Get batch of messages from buffer"""
        batch = []
        start_time = time.time()
        
        while (len(batch) < self.config.batch_size and 
               time.time() - start_time < self.config.max_latency_ms / 1000):
            
            try:
                message = await asyncio.wait_for(self.buffer.get(), timeout=0.1)
                batch.append(message)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_single_message(self, message: StreamMessage):
        """Process single message"""
        try:
            start_time = time.time()
            
            # Apply message handlers
            for handler in self.message_handlers:
                try:
                    result = handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_processed += 1
            self.metrics.total_latency += processing_time
            self.metrics.average_latency = self.metrics.total_latency / self.metrics.messages_processed
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.messages_failed += 1
    
    async def _process_message_batch(self, batch: List[StreamMessage]):
        """Process batch of messages"""
        try:
            start_time = time.time()
            
            # Apply batch handlers
            for handler in self.batch_handlers:
                try:
                    result = handler(batch)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in batch handler: {e}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_processed += len(batch)
            self.metrics.total_latency += processing_time
            self.metrics.average_latency = self.metrics.total_latency / self.metrics.messages_processed
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics.messages_failed += len(batch)
    
    async def _process_message_window(self, window: List[StreamMessage]):
        """Process window of messages"""
        try:
            start_time = time.time()
            
            # Apply window handlers
            for handler in self.window_handlers:
                try:
                    result = handler(window)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in window handler: {e}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_processed += len(window)
            self.metrics.total_latency += processing_time
            self.metrics.average_latency = self.metrics.total_latency / self.metrics.messages_processed
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
            self.metrics.messages_failed += len(window)
    
    async def _checkpoint_manager(self):
        """Checkpoint manager for fault tolerance"""
        logger.info(f"Checkpoint manager started for {self.config.stream_id}")
        
        while self.running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval / 1000)
                await self._create_checkpoint()
                
            except Exception as e:
                logger.error(f"Error in checkpoint manager: {e}")
        
        logger.info(f"Checkpoint manager stopped for {self.config.stream_id}")
    
    async def _create_checkpoint(self):
        """Create processing checkpoint"""
        try:
            checkpoint = {
                'stream_id': self.config.stream_id,
                'timestamp': datetime.now().isoformat(),
                'sequence_number': self.buffer.sequence_counter,
                'buffer_size': self.buffer.size(),
                'metrics': {
                    'messages_received': self.metrics.messages_received,
                    'messages_processed': self.metrics.messages_processed,
                    'messages_failed': self.metrics.messages_failed
                }
            }
            
            self.checkpoint_data = checkpoint
            self.metrics.checkpoint_count += 1
            self.metrics.last_checkpoint = datetime.now()
            
            logger.debug(f"Checkpoint created for {self.config.stream_id}")
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
    
    def get_metrics(self) -> StreamMetrics:
        """Get current stream metrics"""
        # Calculate throughput
        if self.metrics.average_latency > 0:
            self.metrics.throughput_per_second = 1.0 / self.metrics.average_latency
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status"""
        return {
            'stream_id': self.config.stream_id,
            'running': self.running,
            'processing_mode': self.config.processing_mode.value,
            'buffer_size': self.buffer.size(),
            'buffer_capacity': self.buffer.max_size,
            'worker_tasks': len(self.worker_tasks),
            'message_handlers': len(self.message_handlers),
            'batch_handlers': len(self.batch_handlers),
            'window_handlers': len(self.window_handlers),
            'metrics': {
                'messages_received': self.metrics.messages_received,
                'messages_processed': self.metrics.messages_processed,
                'messages_failed': self.metrics.messages_failed,
                'average_latency': self.metrics.average_latency,
                'throughput_per_second': self.metrics.throughput_per_second,
                'backpressure_events': self.metrics.backpressure_events,
                'checkpoint_count': self.metrics.checkpoint_count
            }
        }

class StreamProcessorManager:
    """Manager for multiple stream processors"""
    
    def __init__(self):
        self.processors = {}
        self.running = False
        
    async def start(self):
        """Start all processors"""
        self.running = True
        for processor in self.processors.values():
            await processor.start()
        logger.info("Stream Processor Manager started")
    
    async def stop(self):
        """Stop all processors"""
        self.running = False
        for processor in self.processors.values():
            await processor.stop()
        logger.info("Stream Processor Manager stopped")
    
    def create_processor(self, config: StreamConfig) -> StreamProcessor:
        """Create new stream processor"""
        processor = StreamProcessor(config)
        self.processors[config.stream_id] = processor
        logger.info(f"Created processor for stream {config.stream_id}")
        return processor
    
    def get_processor(self, stream_id: str) -> Optional[StreamProcessor]:
        """Get processor by stream ID"""
        return self.processors.get(stream_id)
    
    def remove_processor(self, stream_id: str) -> bool:
        """Remove processor"""
        if stream_id in self.processors:
            del self.processors[stream_id]
            logger.info(f"Removed processor for stream {stream_id}")
            return True
        return False
    
    def get_all_metrics(self) -> Dict[str, StreamMetrics]:
        """Get metrics for all processors"""
        return {stream_id: processor.get_metrics() 
                for stream_id, processor in self.processors.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_messages = sum(p.metrics.messages_received for p in self.processors.values())
        total_processed = sum(p.metrics.messages_processed for p in self.processors.values())
        total_failed = sum(p.metrics.messages_failed for p in self.processors.values())
        
        return {
            'running': self.running,
            'processor_count': len(self.processors),
            'total_messages_received': total_messages,
            'total_messages_processed': total_processed,
            'total_messages_failed': total_failed,
            'processors': {stream_id: processor.get_status() 
                          for stream_id, processor in self.processors.items()}
        }

# Test and demonstration
async def test_stream_processor():
    """Test the stream processor system"""
    print("🧠 Testing CortexOS Stream Processor...")
    
    # Create stream configurations
    real_time_config = StreamConfig(
        stream_id="neural_stream_rt",
        name="Real-time Neural Stream",
        stream_type=StreamType.NEURAL_SIGNALS,
        processing_mode=ProcessingMode.REAL_TIME,
        buffer_size=1000,
        max_latency_ms=10
    )
    
    batch_config = StreamConfig(
        stream_id="sensor_stream_batch",
        name="Batch Sensor Stream",
        stream_type=StreamType.SENSOR_DATA,
        processing_mode=ProcessingMode.MICRO_BATCH,
        buffer_size=5000,
        batch_size=50,
        max_latency_ms=100
    )
    
    # Create processor manager
    manager = StreamProcessorManager()
    
    # Create processors
    rt_processor = manager.create_processor(real_time_config)
    batch_processor = manager.create_processor(batch_config)
    
    # Add handlers
    def neural_handler(message: StreamMessage):
        print(f"   🧠 Neural signal: {message.data}")
    
    def sensor_batch_handler(batch: List[StreamMessage]):
        print(f"   📊 Sensor batch: {len(batch)} messages")
    
    rt_processor.add_message_handler(neural_handler)
    batch_processor.add_batch_handler(sensor_batch_handler)
    
    try:
        # Start processors
        await manager.start()
        print("✅ Stream processors started")
        
        # Send test messages
        print("\n📤 Sending test messages...")
        
        # Real-time messages
        for i in range(10):
            await rt_processor.send_message(
                data={'signal': f'neural_{i}', 'amplitude': i * 0.1},
                metadata={'source': 'test_neural_sensor'}
            )
        
        # Batch messages
        for i in range(25):
            await batch_processor.send_message(
                data={'sensor_id': f'sensor_{i}', 'value': i * 2.5},
                metadata={'source': 'test_sensor_array'}
            )
        
        # Wait for processing
        print("⏳ Processing messages...")
        await asyncio.sleep(2)
        
        # Display metrics
        print("\n📊 Stream Metrics:")
        all_metrics = manager.get_all_metrics()
        for stream_id, metrics in all_metrics.items():
            print(f"   {stream_id}:")
            print(f"      Received: {metrics.messages_received}")
            print(f"      Processed: {metrics.messages_processed}")
            print(f"      Failed: {metrics.messages_failed}")
            print(f"      Avg Latency: {metrics.average_latency:.6f}s")
            print(f"      Throughput: {metrics.throughput_per_second:.2f} msg/sec")
        
        # Display system status
        print("\n🔧 System Status:")
        status = manager.get_system_status()
        print(f"   Running: {status['running']}")
        print(f"   Processors: {status['processor_count']}")
        print(f"   Total Received: {status['total_messages_received']}")
        print(f"   Total Processed: {status['total_messages_processed']}")
        print(f"   Total Failed: {status['total_messages_failed']}")
        
        print("\n✅ Stream Processor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(test_stream_processor())

