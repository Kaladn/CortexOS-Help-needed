#!/usr/bin/env python3
"""
CortexOS Phase 4: Data Ingestion Engine
Advanced neural data ingestion and preprocessing system
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
import re
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types for ingestion"""
    TEXT = "text"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BINARY = "binary"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"

class IngestionStatus(Enum):
    """Data ingestion status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

@dataclass
class DataSchema:
    """Data schema definition for ingestion"""
    schema_id: str
    name: str
    fields: Dict[str, Dict[str, Any]]
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    transformation_rules: Dict[str, Any] = field(default_factory=dict)
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)

@dataclass
class IngestionJob:
    """Data ingestion job definition"""
    job_id: str
    data_source: str
    schema: DataSchema
    data: Any
    priority: int = 1
    batch_size: int = 1000
    timeout: int = 300
    retry_count: int = 3
    status: IngestionStatus = IngestionStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IngestionMetrics:
    """Data ingestion performance metrics"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    rejected_jobs: int = 0
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0

class DataValidator:
    """Advanced data validation system"""
    
    def __init__(self):
        self.validation_functions = {
            'required': self._validate_required,
            'type': self._validate_type,
            'range': self._validate_range,
            'pattern': self._validate_pattern,
            'length': self._validate_length,
            'unique': self._validate_unique,
            'custom': self._validate_custom
        }
        
    def validate_data(self, data: Dict[str, Any], schema: DataSchema) -> Tuple[bool, List[str]]:
        """Validate data against schema"""
        errors = []
        
        try:
            # Check required fields
            for field in schema.required_fields:
                if field not in data:
                    errors.append(f"Required field '{field}' is missing")
            
            # Validate each field
            for field_name, field_value in data.items():
                if field_name in schema.fields:
                    field_schema = schema.fields[field_name]
                    field_errors = self._validate_field(field_name, field_value, field_schema)
                    errors.extend(field_errors)
            
            # Apply validation rules
            for rule_name, rule_config in schema.validation_rules.items():
                if rule_name in self.validation_functions:
                    rule_errors = self.validation_functions[rule_name](data, rule_config)
                    errors.extend(rule_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False, [f"Validation error: {e}"]
    
    def _validate_field(self, field_name: str, field_value: Any, field_schema: Dict[str, Any]) -> List[str]:
        """Validate individual field"""
        errors = []
        
        try:
            # Type validation
            expected_type = field_schema.get('type')
            if expected_type and not self._check_type(field_value, expected_type):
                errors.append(f"Field '{field_name}' has invalid type. Expected {expected_type}")
            
            # Range validation
            if 'min' in field_schema or 'max' in field_schema:
                if not self._check_range(field_value, field_schema.get('min'), field_schema.get('max')):
                    errors.append(f"Field '{field_name}' is out of range")
            
            # Pattern validation
            if 'pattern' in field_schema:
                if not self._check_pattern(field_value, field_schema['pattern']):
                    errors.append(f"Field '{field_name}' does not match required pattern")
            
            # Length validation
            if 'min_length' in field_schema or 'max_length' in field_schema:
                if not self._check_length(field_value, field_schema.get('min_length'), field_schema.get('max_length')):
                    errors.append(f"Field '{field_name}' has invalid length")
            
            return errors
            
        except Exception as e:
            return [f"Error validating field '{field_name}': {e}"]
    
    def _validate_required(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate required fields"""
        errors = []
        required_fields = config.get('fields', [])
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Required field '{field}' is missing or null")
        
        return errors
    
    def _validate_type(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate data types"""
        errors = []
        type_rules = config.get('rules', {})
        
        for field, expected_type in type_rules.items():
            if field in data and not self._check_type(data[field], expected_type):
                errors.append(f"Field '{field}' has invalid type. Expected {expected_type}")
        
        return errors
    
    def _validate_range(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate numeric ranges"""
        errors = []
        range_rules = config.get('rules', {})
        
        for field, range_config in range_rules.items():
            if field in data:
                value = data[field]
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                if not self._check_range(value, min_val, max_val):
                    errors.append(f"Field '{field}' is out of range [{min_val}, {max_val}]")
        
        return errors
    
    def _validate_pattern(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate string patterns"""
        errors = []
        pattern_rules = config.get('rules', {})
        
        for field, pattern in pattern_rules.items():
            if field in data and not self._check_pattern(data[field], pattern):
                errors.append(f"Field '{field}' does not match pattern '{pattern}'")
        
        return errors
    
    def _validate_length(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate string/array lengths"""
        errors = []
        length_rules = config.get('rules', {})
        
        for field, length_config in length_rules.items():
            if field in data:
                value = data[field]
                min_len = length_config.get('min')
                max_len = length_config.get('max')
                
                if not self._check_length(value, min_len, max_len):
                    errors.append(f"Field '{field}' has invalid length")
        
        return errors
    
    def _validate_unique(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate unique constraints"""
        errors = []
        unique_fields = config.get('fields', [])
        
        # This would require access to existing data for proper uniqueness checking
        # For now, just validate that the field exists
        for field in unique_fields:
            if field not in data:
                errors.append(f"Unique field '{field}' is missing")
        
        return errors
    
    def _validate_custom(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate using custom functions"""
        errors = []
        custom_rules = config.get('rules', {})
        
        for field, rule_config in custom_rules.items():
            if field in data:
                # Custom validation would be implemented here
                # For now, just pass
                pass
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        try:
            if expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'integer':
                return isinstance(value, int)
            elif expected_type == 'float':
                return isinstance(value, (int, float))
            elif expected_type == 'boolean':
                return isinstance(value, bool)
            elif expected_type == 'array':
                return isinstance(value, (list, tuple))
            elif expected_type == 'object':
                return isinstance(value, dict)
            else:
                return True  # Unknown type, assume valid
                
        except Exception:
            return False
    
    def _check_range(self, value: Any, min_val: Optional[float], max_val: Optional[float]) -> bool:
        """Check if value is within range"""
        try:
            if not isinstance(value, (int, float)):
                return True  # Non-numeric values pass range check
            
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_pattern(self, value: Any, pattern: str) -> bool:
        """Check if value matches pattern"""
        try:
            if not isinstance(value, str):
                return True  # Non-string values pass pattern check
            
            return bool(re.match(pattern, value))
            
        except Exception:
            return False
    
    def _check_length(self, value: Any, min_len: Optional[int], max_len: Optional[int]) -> bool:
        """Check if value length is within bounds"""
        try:
            if not hasattr(value, '__len__'):
                return True  # Values without length pass length check
            
            length = len(value)
            
            if min_len is not None and length < min_len:
                return False
            if max_len is not None and length > max_len:
                return False
            
            return True
            
        except Exception:
            return False

class DataTransformer:
    """Advanced data transformation system"""
    
    def __init__(self):
        self.transformation_functions = {
            'normalize': self._normalize,
            'standardize': self._standardize,
            'encode': self._encode,
            'decode': self._decode,
            'convert': self._convert,
            'extract': self._extract,
            'aggregate': self._aggregate,
            'filter': self._filter
        }
    
    def transform_data(self, data: Dict[str, Any], schema: DataSchema) -> Dict[str, Any]:
        """Transform data according to schema"""
        try:
            transformed_data = data.copy()
            
            # Apply field-level transformations
            for field_name, field_value in data.items():
                if field_name in schema.fields:
                    field_schema = schema.fields[field_name]
                    if 'transform' in field_schema:
                        transformed_data[field_name] = self._transform_field(
                            field_value, field_schema['transform']
                        )
            
            # Apply schema-level transformations
            for transform_name, transform_config in schema.transformation_rules.items():
                if transform_name in self.transformation_functions:
                    transformed_data = self.transformation_functions[transform_name](
                        transformed_data, transform_config
                    )
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return data
    
    def _transform_field(self, value: Any, transform_config: Dict[str, Any]) -> Any:
        """Transform individual field"""
        try:
            transform_type = transform_config.get('type')
            
            if transform_type == 'uppercase':
                return str(value).upper() if value is not None else value
            elif transform_type == 'lowercase':
                return str(value).lower() if value is not None else value
            elif transform_type == 'trim':
                return str(value).strip() if value is not None else value
            elif transform_type == 'multiply':
                factor = transform_config.get('factor', 1)
                return value * factor if isinstance(value, (int, float)) else value
            elif transform_type == 'round':
                decimals = transform_config.get('decimals', 0)
                return round(value, decimals) if isinstance(value, (int, float)) else value
            else:
                return value
                
        except Exception as e:
            logger.error(f"Error transforming field: {e}")
            return value
    
    def _normalize(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize numeric fields"""
        try:
            fields = config.get('fields', [])
            min_val = config.get('min', 0)
            max_val = config.get('max', 1)
            
            for field in fields:
                if field in data and isinstance(data[field], (int, float)):
                    # Simple min-max normalization
                    data[field] = min_val + (data[field] - min_val) / (max_val - min_val)
            
            return data
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data
    
    def _standardize(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize numeric fields"""
        try:
            fields = config.get('fields', [])
            
            for field in fields:
                if field in data and isinstance(data[field], (int, float)):
                    # Simple standardization (would need statistics for proper implementation)
                    mean = config.get('mean', 0)
                    std = config.get('std', 1)
                    data[field] = (data[field] - mean) / std
            
            return data
            
        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            return data
    
    def _encode(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical fields"""
        try:
            encoding_rules = config.get('rules', {})
            
            for field, encoding_config in encoding_rules.items():
                if field in data:
                    encoding_type = encoding_config.get('type', 'label')
                    
                    if encoding_type == 'label':
                        # Label encoding
                        mapping = encoding_config.get('mapping', {})
                        data[field] = mapping.get(data[field], data[field])
                    elif encoding_type == 'onehot':
                        # One-hot encoding (simplified)
                        categories = encoding_config.get('categories', [])
                        for category in categories:
                            data[f"{field}_{category}"] = 1 if data[field] == category else 0
            
            return data
            
        except Exception as e:
            logger.error(f"Error encoding data: {e}")
            return data
    
    def _decode(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Decode encoded fields"""
        try:
            decoding_rules = config.get('rules', {})
            
            for field, decoding_config in decoding_rules.items():
                if field in data:
                    # Reverse mapping
                    mapping = decoding_config.get('mapping', {})
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    data[field] = reverse_mapping.get(data[field], data[field])
            
            return data
            
        except Exception as e:
            logger.error(f"Error decoding data: {e}")
            return data
    
    def _convert(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert field types"""
        try:
            conversion_rules = config.get('rules', {})
            
            for field, target_type in conversion_rules.items():
                if field in data:
                    try:
                        if target_type == 'int':
                            data[field] = int(data[field])
                        elif target_type == 'float':
                            data[field] = float(data[field])
                        elif target_type == 'str':
                            data[field] = str(data[field])
                        elif target_type == 'bool':
                            data[field] = bool(data[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert field '{field}' to {target_type}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error converting data: {e}")
            return data
    
    def _extract(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from fields"""
        try:
            extraction_rules = config.get('rules', {})
            
            for field, extraction_config in extraction_rules.items():
                if field in data:
                    extraction_type = extraction_config.get('type')
                    
                    if extraction_type == 'length':
                        data[f"{field}_length"] = len(str(data[field]))
                    elif extraction_type == 'word_count':
                        data[f"{field}_word_count"] = len(str(data[field]).split())
                    elif extraction_type == 'regex':
                        pattern = extraction_config.get('pattern', '')
                        matches = re.findall(pattern, str(data[field]))
                        data[f"{field}_matches"] = len(matches)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return data
    
    def _aggregate(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate fields"""
        try:
            aggregation_rules = config.get('rules', {})
            
            for new_field, agg_config in aggregation_rules.items():
                fields = agg_config.get('fields', [])
                operation = agg_config.get('operation', 'sum')
                
                values = [data.get(field, 0) for field in fields if field in data]
                
                if operation == 'sum':
                    data[new_field] = sum(values)
                elif operation == 'mean':
                    data[new_field] = sum(values) / len(values) if values else 0
                elif operation == 'max':
                    data[new_field] = max(values) if values else 0
                elif operation == 'min':
                    data[new_field] = min(values) if values else 0
            
            return data
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return data
    
    def _filter(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data fields"""
        try:
            filter_rules = config.get('rules', {})
            
            if 'include' in filter_rules:
                # Keep only specified fields
                include_fields = filter_rules['include']
                data = {k: v for k, v in data.items() if k in include_fields}
            elif 'exclude' in filter_rules:
                # Remove specified fields
                exclude_fields = filter_rules['exclude']
                data = {k: v for k, v in data.items() if k not in exclude_fields}
            
            return data
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return data

class DataIngestionEngine:
    """Advanced neural data ingestion engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.schemas = {}
        self.job_queue = asyncio.Queue()
        self.processing_jobs = {}
        self.completed_jobs = {}
        self.metrics = IngestionMetrics()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.running = False
        self.worker_tasks = []
        
        # Configuration
        self.worker_count = self.config.get('worker_count', 4)
        self.max_batch_size = self.config.get('max_batch_size', 10000)
        self.job_timeout = self.config.get('job_timeout', 300)
        self.retry_limit = self.config.get('retry_limit', 3)
        self.enable_validation = self.config.get('enable_validation', True)
        self.enable_transformation = self.config.get('enable_transformation', True)
        
        logger.info("Data Ingestion Engine initialized")
    
    async def start(self):
        """Start the data ingestion engine"""
        try:
            self.running = True
            
            # Start worker tasks
            for i in range(self.worker_count):
                task = asyncio.create_task(self._ingestion_worker(f"worker_{i}"))
                self.worker_tasks.append(task)
            
            logger.info(f"Data Ingestion Engine started with {self.worker_count} workers")
            
        except Exception as e:
            logger.error(f"Error starting Data Ingestion Engine: {e}")
            raise
    
    async def stop(self):
        """Stop the data ingestion engine"""
        try:
            self.running = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            logger.info("Data Ingestion Engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Data Ingestion Engine: {e}")
    
    def register_schema(self, schema: DataSchema) -> bool:
        """Register data schema for ingestion"""
        try:
            self.schemas[schema.schema_id] = schema
            logger.info(f"Registered schema '{schema.schema_id}': {schema.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering schema: {e}")
            return False
    
    async def submit_job(self, job: IngestionJob) -> str:
        """Submit data ingestion job"""
        try:
            # Validate job
            if job.schema.schema_id not in self.schemas:
                raise ValueError(f"Unknown schema: {job.schema.schema_id}")
            
            # Add to queue
            await self.job_queue.put(job)
            self.metrics.total_jobs += 1
            
            logger.info(f"Submitted ingestion job {job.job_id}")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Get status of ingestion job"""
        try:
            # Check processing jobs
            if job_id in self.processing_jobs:
                return self.processing_jobs[job_id]
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    async def _ingestion_worker(self, worker_id: str):
        """Worker task for processing ingestion jobs"""
        logger.info(f"Ingestion worker {worker_id} started")
        
        while self.running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Process job
                await self._process_ingestion_job(job, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in ingestion worker {worker_id}: {e}")
        
        logger.info(f"Ingestion worker {worker_id} stopped")
    
    async def _process_ingestion_job(self, job: IngestionJob, worker_id: str):
        """Process a single ingestion job"""
        try:
            start_time = time.time()
            job.status = IngestionStatus.PROCESSING
            job.started_time = datetime.now()
            self.processing_jobs[job.job_id] = job
            
            logger.info(f"Worker {worker_id} processing job {job.job_id}")
            
            # Get schema
            schema = self.schemas[job.schema.schema_id]
            
            # Process data based on type
            if isinstance(job.data, list):
                # Batch processing
                processed_data = await self._process_batch_data(job.data, schema, job)
            elif isinstance(job.data, dict):
                # Single record processing
                processed_data = await self._process_single_record(job.data, schema, job)
            else:
                raise ValueError(f"Unsupported data type: {type(job.data)}")
            
            # Update job status
            job.status = IngestionStatus.COMPLETED
            job.completed_time = datetime.now()
            job.metadata['processed_data'] = processed_data
            job.metadata['processing_time'] = time.time() - start_time
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.processing_jobs:
                del self.processing_jobs[job.job_id]
            
            # Update metrics
            self.metrics.completed_jobs += 1
            processing_time = time.time() - start_time
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (self.metrics.completed_jobs - 1) + processing_time)
                / self.metrics.completed_jobs
            )
            
            if isinstance(job.data, list):
                self.metrics.total_records += len(job.data)
                self.metrics.processed_records += len(processed_data)
            else:
                self.metrics.total_records += 1
                self.metrics.processed_records += 1
            
            logger.info(f"Completed job {job.job_id} in {processing_time:.3f}s")
            
        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_time = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.processing_jobs:
                del self.processing_jobs[job.job_id]
            
            self.metrics.failed_jobs += 1
            logger.error(f"Failed job {job.job_id}: {e}")
    
    async def _process_batch_data(self, data: List[Dict[str, Any]], 
                                schema: DataSchema, job: IngestionJob) -> List[Dict[str, Any]]:
        """Process batch data"""
        processed_records = []
        failed_records = []
        
        try:
            # Process in chunks
            chunk_size = min(job.batch_size, self.max_batch_size)
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                for record in chunk:
                    try:
                        processed_record = await self._process_single_record(record, schema, job)
                        if processed_record is not None:
                            processed_records.append(processed_record)
                    except Exception as e:
                        failed_records.append({
                            'record': record,
                            'error': str(e)
                        })
                        self.metrics.failed_records += 1
                
                # Allow other tasks to run
                await asyncio.sleep(0)
            
            # Store failed records in job metadata
            if failed_records:
                job.metadata['failed_records'] = failed_records
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Error processing batch data: {e}")
            raise
    
    async def _process_single_record(self, record: Dict[str, Any], 
                                   schema: DataSchema, job: IngestionJob) -> Optional[Dict[str, Any]]:
        """Process single data record"""
        try:
            # Validation
            if self.enable_validation:
                is_valid, errors = self.validator.validate_data(record, schema)
                if not is_valid:
                    logger.warning(f"Validation failed for record: {errors}")
                    return None
            
            # Transformation
            if self.enable_transformation:
                record = self.transformer.transform_data(record, schema)
            
            # Add metadata
            record['_ingestion_timestamp'] = datetime.now().isoformat()
            record['_job_id'] = job.job_id
            record['_schema_id'] = schema.schema_id
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            raise
    
    def get_metrics(self) -> IngestionMetrics:
        """Get current ingestion metrics"""
        # Calculate derived metrics
        if self.metrics.total_jobs > 0:
            self.metrics.error_rate = self.metrics.failed_jobs / self.metrics.total_jobs
        
        if self.metrics.average_processing_time > 0:
            self.metrics.throughput_per_second = 1.0 / self.metrics.average_processing_time
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'registered_schemas': len(self.schemas),
            'pending_jobs': self.job_queue.qsize(),
            'processing_jobs': len(self.processing_jobs),
            'completed_jobs': len(self.completed_jobs),
            'worker_count': len(self.worker_tasks),
            'metrics': {
                'total_jobs': self.metrics.total_jobs,
                'completed_jobs': self.metrics.completed_jobs,
                'failed_jobs': self.metrics.failed_jobs,
                'total_records': self.metrics.total_records,
                'processed_records': self.metrics.processed_records,
                'failed_records': self.metrics.failed_records,
                'average_processing_time': self.metrics.average_processing_time,
                'throughput_per_second': self.metrics.throughput_per_second,
                'error_rate': self.metrics.error_rate
            }
        }

# Test and demonstration
async def test_data_ingestion_engine():
    """Test the data ingestion engine"""
    print("🧠 Testing CortexOS Data Ingestion Engine...")
    
    # Initialize engine
    config = {
        'worker_count': 2,
        'max_batch_size': 1000,
        'job_timeout': 60,
        'enable_validation': True,
        'enable_transformation': True
    }
    
    engine = DataIngestionEngine(config)
    await engine.start()
    
    try:
        # Create test schema
        schema = DataSchema(
            schema_id="test_schema",
            name="Test Data Schema",
            fields={
                'id': {'type': 'integer', 'min': 1},
                'name': {'type': 'string', 'min_length': 1, 'max_length': 100},
                'value': {'type': 'float', 'min': 0.0, 'max': 100.0},
                'category': {'type': 'string', 'pattern': r'^[A-Z]+$'}
            },
            required_fields={'id', 'name'},
            validation_rules={
                'required': {'fields': ['id', 'name']},
                'type': {'rules': {'id': 'integer', 'name': 'string'}}
            },
            transformation_rules={
                'normalize': {'fields': ['value'], 'min': 0, 'max': 100},
                'convert': {'rules': {'id': 'int'}}
            }
        )
        
        # Register schema
        success = engine.register_schema(schema)
        print(f"📋 Schema registration: {'✅' if success else '❌'}")
        
        # Create test data
        test_data = [
            {'id': 1, 'name': 'Test Item 1', 'value': 75.5, 'category': 'ALPHA'},
            {'id': 2, 'name': 'Test Item 2', 'value': 42.3, 'category': 'BETA'},
            {'id': 3, 'name': 'Test Item 3', 'value': 88.9, 'category': 'GAMMA'},
            {'id': '4', 'name': 'Test Item 4', 'value': '65.2', 'category': 'DELTA'},  # Type conversion test
            {'id': 5, 'name': '', 'value': 150.0, 'category': 'invalid'},  # Validation failure test
        ]
        
        # Submit batch job
        batch_job = IngestionJob(
            job_id="batch_job_001",
            data_source="test_batch",
            schema=schema,
            data=test_data,
            batch_size=100
        )
        
        print("📤 Submitting batch ingestion job...")
        job_id = await engine.submit_job(batch_job)
        print(f"   Job ID: {job_id}")
        
        # Submit single record job
        single_record = {'id': 6, 'name': 'Single Test Item', 'value': 33.7, 'category': 'OMEGA'}
        single_job = IngestionJob(
            job_id="single_job_001",
            data_source="test_single",
            schema=schema,
            data=single_record
        )
        
        print("📤 Submitting single record ingestion job...")
        single_job_id = await engine.submit_job(single_job)
        print(f"   Job ID: {single_job_id}")
        
        # Wait for processing
        print("\n⏳ Waiting for job processing...")
        await asyncio.sleep(3)
        
        # Check job statuses
        print("\n📊 Job Status:")
        batch_status = await engine.get_job_status(job_id)
        if batch_status:
            print(f"   Batch Job: {batch_status.status.value}")
            if batch_status.status == IngestionStatus.COMPLETED:
                processed_count = len(batch_status.metadata.get('processed_data', []))
                failed_count = len(batch_status.metadata.get('failed_records', []))
                print(f"      Processed: {processed_count} records")
                print(f"      Failed: {failed_count} records")
        
        single_status = await engine.get_job_status(single_job_id)
        if single_status:
            print(f"   Single Job: {single_status.status.value}")
        
        # Display metrics
        print("\n📈 Ingestion Metrics:")
        metrics = engine.get_metrics()
        print(f"   Total jobs: {metrics.total_jobs}")
        print(f"   Completed jobs: {metrics.completed_jobs}")
        print(f"   Failed jobs: {metrics.failed_jobs}")
        print(f"   Total records: {metrics.total_records}")
        print(f"   Processed records: {metrics.processed_records}")
        print(f"   Failed records: {metrics.failed_records}")
        print(f"   Average processing time: {metrics.average_processing_time:.3f}s")
        print(f"   Throughput: {metrics.throughput_per_second:.2f} jobs/sec")
        print(f"   Error rate: {metrics.error_rate:.3f}")
        
        # Display status
        print("\n🔧 System Status:")
        status = engine.get_status()
        for key, value in status.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
        print("\n✅ Data Ingestion Engine test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_data_ingestion_engine())

