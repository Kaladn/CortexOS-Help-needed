#!/usr/bin/env python3
"""
CortexOS Phase 4: Ingestion Validator
Advanced data validation and quality assurance system
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
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    CUSTOM = "custom"

class ValidationResult(Enum):
    """Validation result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # 'format', 'range', 'pattern', 'custom', 'statistical'
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # 'error', 'warning', 'info'
    enabled: bool = True
    custom_function: Optional[Callable] = None

@dataclass
class ValidationSchema:
    """Validation schema for data types"""
    schema_id: str
    name: str
    description: str
    rules: List[ValidationRule] = field(default_factory=list)
    field_rules: Dict[str, List[ValidationRule]] = field(default_factory=dict)
    global_rules: List[ValidationRule] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.MODERATE

@dataclass
class ValidationError:
    """Validation error details"""
    error_id: str
    rule_id: str
    field_name: Optional[str]
    error_message: str
    severity: str
    value: Any = None
    expected: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    schema_id: str
    data_id: str
    validation_time: datetime
    total_records: int
    valid_records: int
    invalid_records: int
    warning_records: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    quality_score: float = 0.0
    quality_level: DataQuality = DataQuality.FAIR
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationMetrics:
    """Validation system metrics"""
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    warning_validations: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    error_rate: float = 0.0
    throughput_per_second: float = 0.0

class StatisticalValidator:
    """Statistical validation methods"""
    
    @staticmethod
    def detect_outliers(values: List[float], method: str = "iqr", threshold: float = 1.5) -> List[int]:
        """Detect outliers in numeric data"""
        if not values or len(values) < 3:
            return []
        
        try:
            if method == "iqr":
                q1 = statistics.quantiles(values, n=4)[0]
                q3 = statistics.quantiles(values, n=4)[2]
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers = []
                for i, value in enumerate(values):
                    if value < lower_bound or value > upper_bound:
                        outliers.append(i)
                return outliers
                
            elif method == "zscore":
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val == 0:
                    return []
                
                outliers = []
                for i, value in enumerate(values):
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > threshold:
                        outliers.append(i)
                return outliers
            
            return []
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []
    
    @staticmethod
    def check_distribution(values: List[float], expected_distribution: str = "normal") -> Dict[str, Any]:
        """Check if data follows expected distribution"""
        if not values or len(values) < 10:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        try:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Basic distribution checks
            if expected_distribution == "normal":
                # Simple normality check using skewness and kurtosis approximation
                sorted_values = sorted(values)
                median_val = statistics.median(values)
                
                # Check if mean ≈ median (normal distribution property)
                mean_median_diff = abs(mean_val - median_val) / std_val if std_val > 0 else 0
                
                return {
                    'valid': mean_median_diff < 0.5,
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'mean_median_ratio': mean_median_diff
                }
            
            return {'valid': True, 'mean': mean_val, 'std': std_val}
            
        except Exception as e:
            logger.error(f"Error checking distribution: {e}")
            return {'valid': False, 'reason': str(e)}
    
    @staticmethod
    def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation between two numeric series"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            sum_y2 = sum(y * y for y in y_values)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

class RuleEngine:
    """Validation rule execution engine"""
    
    def __init__(self):
        self.rule_executors = {
            'format': self._execute_format_rule,
            'range': self._execute_range_rule,
            'pattern': self._execute_pattern_rule,
            'length': self._execute_length_rule,
            'type': self._execute_type_rule,
            'required': self._execute_required_rule,
            'unique': self._execute_unique_rule,
            'custom': self._execute_custom_rule,
            'statistical': self._execute_statistical_rule
        }
    
    def execute_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any] = None) -> Tuple[ValidationResult, Optional[str]]:
        """Execute validation rule"""
        try:
            if not rule.enabled:
                return ValidationResult.SKIP, "Rule disabled"
            
            if rule.rule_type in self.rule_executors:
                return self.rule_executors[rule.rule_type](rule, value, context or {})
            else:
                return ValidationResult.FAIL, f"Unknown rule type: {rule.rule_type}"
                
        except Exception as e:
            logger.error(f"Error executing rule {rule.rule_id}: {e}")
            return ValidationResult.FAIL, f"Rule execution error: {e}"
    
    def _execute_format_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute format validation rule"""
        format_type = rule.parameters.get('format_type')
        
        if format_type == 'email':
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(pattern, str(value)):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, "Invalid email format"
        
        elif format_type == 'phone':
            pattern = r'^\+?1?-?\.?\s?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$'
            if re.match(pattern, str(value)):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, "Invalid phone format"
        
        elif format_type == 'date':
            date_format = rule.parameters.get('date_format', '%Y-%m-%d')
            try:
                datetime.strptime(str(value), date_format)
                return ValidationResult.PASS, None
            except ValueError:
                return ValidationResult.FAIL, f"Invalid date format, expected {date_format}"
        
        elif format_type == 'url':
            pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
            if re.match(pattern, str(value)):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, "Invalid URL format"
        
        return ValidationResult.FAIL, f"Unknown format type: {format_type}"
    
    def _execute_range_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute range validation rule"""
        try:
            numeric_value = float(value)
            min_val = rule.parameters.get('min')
            max_val = rule.parameters.get('max')
            
            if min_val is not None and numeric_value < min_val:
                return ValidationResult.FAIL, f"Value {numeric_value} is below minimum {min_val}"
            
            if max_val is not None and numeric_value > max_val:
                return ValidationResult.FAIL, f"Value {numeric_value} is above maximum {max_val}"
            
            return ValidationResult.PASS, None
            
        except (ValueError, TypeError):
            return ValidationResult.FAIL, "Value is not numeric"
    
    def _execute_pattern_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute pattern validation rule"""
        pattern = rule.parameters.get('pattern')
        if not pattern:
            return ValidationResult.FAIL, "No pattern specified"
        
        try:
            if re.match(pattern, str(value)):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Value does not match pattern {pattern}"
            
        except re.error as e:
            return ValidationResult.FAIL, f"Invalid regex pattern: {e}"
    
    def _execute_length_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute length validation rule"""
        try:
            length = len(value) if hasattr(value, '__len__') else len(str(value))
            min_len = rule.parameters.get('min_length')
            max_len = rule.parameters.get('max_length')
            
            if min_len is not None and length < min_len:
                return ValidationResult.FAIL, f"Length {length} is below minimum {min_len}"
            
            if max_len is not None and length > max_len:
                return ValidationResult.FAIL, f"Length {length} is above maximum {max_len}"
            
            return ValidationResult.PASS, None
            
        except Exception as e:
            return ValidationResult.FAIL, f"Error checking length: {e}"
    
    def _execute_type_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute type validation rule"""
        expected_type = rule.parameters.get('expected_type')
        
        if expected_type == 'string':
            if isinstance(value, str):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected string, got {type(value).__name__}"
        
        elif expected_type == 'integer':
            if isinstance(value, int) and not isinstance(value, bool):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected integer, got {type(value).__name__}"
        
        elif expected_type == 'float':
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected numeric, got {type(value).__name__}"
        
        elif expected_type == 'boolean':
            if isinstance(value, bool):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected boolean, got {type(value).__name__}"
        
        elif expected_type == 'list':
            if isinstance(value, list):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected list, got {type(value).__name__}"
        
        elif expected_type == 'dict':
            if isinstance(value, dict):
                return ValidationResult.PASS, None
            return ValidationResult.FAIL, f"Expected dict, got {type(value).__name__}"
        
        return ValidationResult.FAIL, f"Unknown expected type: {expected_type}"
    
    def _execute_required_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute required field validation rule"""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return ValidationResult.FAIL, "Required field is missing or empty"
        return ValidationResult.PASS, None
    
    def _execute_unique_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute uniqueness validation rule"""
        # This would require access to existing data for proper uniqueness checking
        # For now, just check if value exists
        existing_values = context.get('existing_values', set())
        
        if value in existing_values:
            return ValidationResult.FAIL, f"Value {value} is not unique"
        
        return ValidationResult.PASS, None
    
    def _execute_custom_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute custom validation rule"""
        if not rule.custom_function:
            return ValidationResult.FAIL, "No custom function provided"
        
        try:
            result = rule.custom_function(value, rule.parameters, context)
            
            if isinstance(result, bool):
                return ValidationResult.PASS if result else ValidationResult.FAIL, None
            elif isinstance(result, tuple) and len(result) == 2:
                return result
            else:
                return ValidationResult.FAIL, "Invalid custom function return value"
                
        except Exception as e:
            return ValidationResult.FAIL, f"Custom function error: {e}"
    
    def _execute_statistical_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> Tuple[ValidationResult, Optional[str]]:
        """Execute statistical validation rule"""
        stat_type = rule.parameters.get('stat_type')
        
        if stat_type == 'outlier_detection':
            # This would require a dataset context
            dataset = context.get('dataset', [])
            if not dataset:
                return ValidationResult.SKIP, "No dataset for outlier detection"
            
            try:
                numeric_value = float(value)
                numeric_dataset = [float(x) for x in dataset if isinstance(x, (int, float))]
                
                outliers = StatisticalValidator.detect_outliers(numeric_dataset)
                value_index = context.get('value_index', -1)
                
                if value_index in outliers:
                    return ValidationResult.WARNING, f"Value {numeric_value} is a statistical outlier"
                
                return ValidationResult.PASS, None
                
            except (ValueError, TypeError):
                return ValidationResult.FAIL, "Value is not numeric for statistical analysis"
        
        return ValidationResult.FAIL, f"Unknown statistical rule type: {stat_type}"

class IngestionValidator:
    """Advanced data validation and quality assurance system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.schemas = {}
        self.rule_engine = RuleEngine()
        self.statistical_validator = StatisticalValidator()
        self.metrics = ValidationMetrics()
        self.validation_cache = {}
        
        # Configuration
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        self.enable_statistical_validation = self.config.get('enable_statistical_validation', True)
        
        logger.info("Ingestion Validator initialized")
    
    def register_schema(self, schema: ValidationSchema) -> bool:
        """Register validation schema"""
        try:
            self.schemas[schema.schema_id] = schema
            logger.info(f"Registered validation schema: {schema.schema_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering schema: {e}")
            return False
    
    async def validate_data(self, data: Any, schema_id: str, data_id: str = None) -> ValidationReport:
        """Validate data against schema"""
        try:
            start_time = time.time()
            data_id = data_id or f"data_{int(time.time() * 1000000)}"
            
            # Get schema
            if schema_id not in self.schemas:
                raise ValueError(f"Unknown schema: {schema_id}")
            
            schema = self.schemas[schema_id]
            
            # Check cache
            cache_key = self._generate_cache_key(data, schema_id)
            if self.enable_caching and cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.debug(f"Cache hit for validation {data_id}")
                    return cached_result['report']
            
            # Perform validation
            if isinstance(data, list):
                report = await self._validate_batch_data(data, schema, data_id)
            elif isinstance(data, dict):
                report = await self._validate_single_record(data, schema, data_id)
            else:
                report = await self._validate_single_value(data, schema, data_id)
            
            # Calculate processing time
            report.processing_time = time.time() - start_time
            
            # Calculate quality metrics
            self._calculate_quality_metrics(report)
            
            # Update system metrics
            self._update_metrics(report)
            
            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, report)
            
            logger.info(f"Validated data {data_id}: {report.quality_level.value} quality")
            return report
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            # Return error report
            return ValidationReport(
                report_id=f"error_{int(time.time() * 1000000)}",
                schema_id=schema_id,
                data_id=data_id or "unknown",
                validation_time=datetime.now(),
                total_records=0,
                valid_records=0,
                invalid_records=1,
                warning_records=0,
                errors=[ValidationError(
                    error_id=f"system_error_{int(time.time() * 1000000)}",
                    rule_id="system",
                    field_name=None,
                    error_message=str(e),
                    severity="error"
                )],
                quality_level=DataQuality.UNACCEPTABLE
            )
    
    async def _validate_batch_data(self, data: List[Any], schema: ValidationSchema, data_id: str) -> ValidationReport:
        """Validate batch of data records"""
        report = ValidationReport(
            report_id=f"batch_{int(time.time() * 1000000)}",
            schema_id=schema.schema_id,
            data_id=data_id,
            validation_time=datetime.now(),
            total_records=len(data),
            valid_records=0,
            invalid_records=0,
            warning_records=0
        )
        
        # Validate each record
        for i, record in enumerate(data):
            record_errors, record_warnings = await self._validate_record(record, schema, i)
            
            if record_errors:
                report.invalid_records += 1
                report.errors.extend(record_errors)
            elif record_warnings:
                report.warning_records += 1
                report.warnings.extend(record_warnings)
            else:
                report.valid_records += 1
        
        # Perform batch-level statistical validation if enabled
        if self.enable_statistical_validation:
            await self._perform_statistical_validation(data, schema, report)
        
        return report
    
    async def _validate_single_record(self, data: Dict[str, Any], schema: ValidationSchema, data_id: str) -> ValidationReport:
        """Validate single data record"""
        report = ValidationReport(
            report_id=f"single_{int(time.time() * 1000000)}",
            schema_id=schema.schema_id,
            data_id=data_id,
            validation_time=datetime.now(),
            total_records=1,
            valid_records=0,
            invalid_records=0,
            warning_records=0
        )
        
        # Validate record
        record_errors, record_warnings = await self._validate_record(data, schema, 0)
        
        if record_errors:
            report.invalid_records = 1
            report.errors.extend(record_errors)
        elif record_warnings:
            report.warning_records = 1
            report.warnings.extend(record_warnings)
        else:
            report.valid_records = 1
        
        return report
    
    async def _validate_single_value(self, data: Any, schema: ValidationSchema, data_id: str) -> ValidationReport:
        """Validate single value"""
        report = ValidationReport(
            report_id=f"value_{int(time.time() * 1000000)}",
            schema_id=schema.schema_id,
            data_id=data_id,
            validation_time=datetime.now(),
            total_records=1,
            valid_records=0,
            invalid_records=0,
            warning_records=0
        )
        
        # Apply global rules to the value
        errors = []
        warnings = []
        
        for rule in schema.global_rules:
            result, message = self.rule_engine.execute_rule(rule, data)
            
            if result == ValidationResult.FAIL:
                error = ValidationError(
                    error_id=f"error_{int(time.time() * 1000000)}",
                    rule_id=rule.rule_id,
                    field_name=None,
                    error_message=message or "Validation failed",
                    severity=rule.severity,
                    value=data
                )
                
                if rule.severity == "error":
                    errors.append(error)
                else:
                    warnings.append(error)
        
        if errors:
            report.invalid_records = 1
            report.errors.extend(errors)
        elif warnings:
            report.warning_records = 1
            report.warnings.extend(warnings)
        else:
            report.valid_records = 1
        
        return report
    
    async def _validate_record(self, record: Dict[str, Any], schema: ValidationSchema, record_index: int) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate individual record"""
        errors = []
        warnings = []
        
        # Validate each field
        for field_name, field_value in record.items():
            if field_name in schema.field_rules:
                field_rules = schema.field_rules[field_name]
                
                for rule in field_rules:
                    context = {
                        'record': record,
                        'record_index': record_index,
                        'field_name': field_name
                    }
                    
                    result, message = self.rule_engine.execute_rule(rule, field_value, context)
                    
                    if result == ValidationResult.FAIL:
                        error = ValidationError(
                            error_id=f"error_{int(time.time() * 1000000)}",
                            rule_id=rule.rule_id,
                            field_name=field_name,
                            error_message=message or "Validation failed",
                            severity=rule.severity,
                            value=field_value
                        )
                        
                        if rule.severity == "error":
                            errors.append(error)
                        else:
                            warnings.append(error)
        
        # Apply global rules
        for rule in schema.global_rules:
            context = {
                'record': record,
                'record_index': record_index
            }
            
            result, message = self.rule_engine.execute_rule(rule, record, context)
            
            if result == ValidationResult.FAIL:
                error = ValidationError(
                    error_id=f"error_{int(time.time() * 1000000)}",
                    rule_id=rule.rule_id,
                    field_name=None,
                    error_message=message or "Validation failed",
                    severity=rule.severity,
                    value=record
                )
                
                if rule.severity == "error":
                    errors.append(error)
                else:
                    warnings.append(error)
        
        return errors, warnings
    
    async def _perform_statistical_validation(self, data: List[Any], schema: ValidationSchema, report: ValidationReport):
        """Perform statistical validation on batch data"""
        try:
            # Extract numeric fields for statistical analysis
            numeric_fields = {}
            
            for record in data:
                if isinstance(record, dict):
                    for field_name, field_value in record.items():
                        if isinstance(field_value, (int, float)):
                            if field_name not in numeric_fields:
                                numeric_fields[field_name] = []
                            numeric_fields[field_name].append(field_value)
            
            # Perform outlier detection
            for field_name, values in numeric_fields.items():
                if len(values) >= 10:  # Minimum sample size for statistical analysis
                    outlier_indices = self.statistical_validator.detect_outliers(values)
                    
                    for outlier_index in outlier_indices:
                        warning = ValidationError(
                            error_id=f"outlier_{int(time.time() * 1000000)}",
                            rule_id="statistical_outlier",
                            field_name=field_name,
                            error_message=f"Statistical outlier detected: {values[outlier_index]}",
                            severity="warning",
                            value=values[outlier_index]
                        )
                        report.warnings.append(warning)
            
        except Exception as e:
            logger.error(f"Error in statistical validation: {e}")
    
    def _calculate_quality_metrics(self, report: ValidationReport):
        """Calculate data quality metrics"""
        try:
            if report.total_records == 0:
                report.quality_score = 0.0
                report.quality_level = DataQuality.UNACCEPTABLE
                return
            
            # Calculate base quality score
            valid_ratio = report.valid_records / report.total_records
            warning_ratio = report.warning_records / report.total_records
            error_ratio = report.invalid_records / report.total_records
            
            # Quality score calculation
            quality_score = valid_ratio * 1.0 + warning_ratio * 0.7 + error_ratio * 0.0
            
            # Adjust for severity of errors
            critical_errors = sum(1 for error in report.errors if error.severity == "error")
            if critical_errors > 0:
                quality_score *= (1.0 - min(critical_errors / report.total_records, 0.5))
            
            report.quality_score = max(0.0, min(1.0, quality_score))
            
            # Determine quality level
            if report.quality_score >= 0.9:
                report.quality_level = DataQuality.EXCELLENT
            elif report.quality_score >= 0.8:
                report.quality_level = DataQuality.GOOD
            elif report.quality_score >= 0.6:
                report.quality_level = DataQuality.FAIR
            elif report.quality_score >= 0.4:
                report.quality_level = DataQuality.POOR
            else:
                report.quality_level = DataQuality.UNACCEPTABLE
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            report.quality_score = 0.0
            report.quality_level = DataQuality.UNACCEPTABLE
    
    def _update_metrics(self, report: ValidationReport):
        """Update system metrics"""
        try:
            self.metrics.total_validations += 1
            
            if report.quality_level in [DataQuality.EXCELLENT, DataQuality.GOOD]:
                self.metrics.passed_validations += 1
            elif report.quality_level in [DataQuality.POOR, DataQuality.UNACCEPTABLE]:
                self.metrics.failed_validations += 1
            else:
                self.metrics.warning_validations += 1
            
            # Update average processing time
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (self.metrics.total_validations - 1) + report.processing_time)
                / self.metrics.total_validations
            )
            
            # Update average quality score
            self.metrics.average_quality_score = (
                (self.metrics.average_quality_score * (self.metrics.total_validations - 1) + report.quality_score)
                / self.metrics.total_validations
            )
            
            # Calculate error rate
            self.metrics.error_rate = self.metrics.failed_validations / self.metrics.total_validations
            
            # Calculate throughput
            if self.metrics.average_processing_time > 0:
                self.metrics.throughput_per_second = 1.0 / self.metrics.average_processing_time
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _generate_cache_key(self, data: Any, schema_id: str) -> str:
        """Generate cache key for validation result"""
        try:
            data_hash = hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            return f"{schema_id}_{data_hash}"
        except Exception:
            return f"{schema_id}_{int(time.time() * 1000000)}"
    
    def _cache_result(self, cache_key: str, report: ValidationReport):
        """Cache validation result"""
        try:
            # Clean old cache entries if needed
            if len(self.validation_cache) >= self.max_cache_size:
                self._clean_cache()
            
            self.validation_cache[cache_key] = {
                'report': report,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def _clean_cache(self):
        """Clean old cache entries"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.validation_cache.items():
                if current_time - entry['timestamp'] > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.validation_cache[key]
            
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def get_metrics(self) -> ValidationMetrics:
        """Get current validation metrics"""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'registered_schemas': len(self.schemas),
            'cache_size': len(self.validation_cache),
            'enable_caching': self.enable_caching,
            'enable_statistical_validation': self.enable_statistical_validation,
            'metrics': {
                'total_validations': self.metrics.total_validations,
                'passed_validations': self.metrics.passed_validations,
                'failed_validations': self.metrics.failed_validations,
                'warning_validations': self.metrics.warning_validations,
                'average_processing_time': self.metrics.average_processing_time,
                'average_quality_score': self.metrics.average_quality_score,
                'error_rate': self.metrics.error_rate,
                'throughput_per_second': self.metrics.throughput_per_second
            }
        }

# Test and demonstration
async def test_ingestion_validator():
    """Test the ingestion validator system"""
    print("🧠 Testing CortexOS Ingestion Validator...")
    
    # Initialize validator
    config = {
        'enable_caching': True,
        'cache_ttl': 300,
        'max_cache_size': 1000,
        'enable_statistical_validation': True
    }
    
    validator = IngestionValidator(config)
    
    # Create validation rules
    email_rule = ValidationRule(
        rule_id="email_format",
        name="Email Format Validation",
        description="Validate email format",
        rule_type="format",
        parameters={'format_type': 'email'},
        severity="error"
    )
    
    age_rule = ValidationRule(
        rule_id="age_range",
        name="Age Range Validation",
        description="Validate age is between 0 and 150",
        rule_type="range",
        parameters={'min': 0, 'max': 150},
        severity="error"
    )
    
    name_rule = ValidationRule(
        rule_id="name_required",
        name="Name Required",
        description="Name field is required",
        rule_type="required",
        severity="error"
    )
    
    # Create validation schema
    schema = ValidationSchema(
        schema_id="user_data",
        name="User Data Validation",
        description="Validation schema for user data",
        field_rules={
            'email': [email_rule],
            'age': [age_rule],
            'name': [name_rule]
        },
        validation_level=ValidationLevel.STRICT
    )
    
    # Register schema
    success = validator.register_schema(schema)
    print(f"📋 Schema registration: {'✅' if success else '❌'}")
    
    try:
        # Test data
        test_data = [
            {'name': 'John Doe', 'email': 'john@example.com', 'age': 30},
            {'name': 'Jane Smith', 'email': 'jane@example.com', 'age': 25},
            {'name': 'Bob Johnson', 'email': 'invalid-email', 'age': 35},  # Invalid email
            {'name': '', 'email': 'empty@example.com', 'age': 40},  # Empty name
            {'name': 'Old Person', 'email': 'old@example.com', 'age': 200},  # Invalid age
        ]
        
        # Validate batch data
        print("\n📊 Validating batch data...")
        batch_report = await validator.validate_data(test_data, "user_data", "batch_001")
        
        print(f"   Total records: {batch_report.total_records}")
        print(f"   Valid records: {batch_report.valid_records}")
        print(f"   Invalid records: {batch_report.invalid_records}")
        print(f"   Warning records: {batch_report.warning_records}")
        print(f"   Quality score: {batch_report.quality_score:.3f}")
        print(f"   Quality level: {batch_report.quality_level.value}")
        print(f"   Processing time: {batch_report.processing_time:.3f}s")
        
        # Display errors
        if batch_report.errors:
            print("\n❌ Validation Errors:")
            for error in batch_report.errors[:5]:  # Show first 5 errors
                print(f"      {error.field_name}: {error.error_message}")
        
        # Validate single record
        print("\n📝 Validating single record...")
        single_record = {'name': 'Test User', 'email': 'test@example.com', 'age': 28}
        single_report = await validator.validate_data(single_record, "user_data", "single_001")
        
        print(f"   Quality score: {single_report.quality_score:.3f}")
        print(f"   Quality level: {single_report.quality_level.value}")
        print(f"   Valid: {'✅' if single_report.valid_records > 0 else '❌'}")
        
        # Display metrics
        print("\n📈 Validation Metrics:")
        metrics = validator.get_metrics()
        print(f"   Total validations: {metrics.total_validations}")
        print(f"   Passed validations: {metrics.passed_validations}")
        print(f"   Failed validations: {metrics.failed_validations}")
        print(f"   Warning validations: {metrics.warning_validations}")
        print(f"   Average processing time: {metrics.average_processing_time:.3f}s")
        print(f"   Average quality score: {metrics.average_quality_score:.3f}")
        print(f"   Error rate: {metrics.error_rate:.3f}")
        print(f"   Throughput: {metrics.throughput_per_second:.2f} validations/sec")
        
        # Display status
        print("\n🔧 System Status:")
        status = validator.get_status()
        for key, value in status.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
        print("\n✅ Ingestion Validator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ingestion_validator())

