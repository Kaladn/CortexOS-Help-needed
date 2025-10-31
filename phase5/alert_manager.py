#!/usr/bin/env python3
"""
CortexOS Phase 5: Alert Manager
Advanced alerting and notification system
"""

import asyncio
import logging
import time
import threading
import smtplib
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status types"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"
    SLACK = "slack"
    SMS = "sms"

class EscalationLevel(Enum):
    """Alert escalation levels"""
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    EXECUTIVE = "executive"

@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    condition: str  # Condition expression
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 15
    auto_resolve: bool = True
    auto_resolve_timeout: int = 3600  # seconds
    escalation_enabled: bool = False
    escalation_timeout: int = 1800  # 30 minutes
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Individual alert instance"""
    alert_id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    source: str
    timestamp: datetime
    resolved_timestamp: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_timestamp: Optional[datetime] = None
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalated_timestamp: Optional[datetime] = None
    notification_count: int = 0
    last_notification: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class NotificationTarget:
    """Notification target configuration"""
    target_id: str
    name: str
    channel: NotificationChannel
    address: str  # email, webhook URL, etc.
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    escalation_levels: List[EscalationLevel] = field(default_factory=list)
    quiet_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)
    rate_limit: int = 10  # max notifications per hour
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationPolicy:
    """Alert escalation policy"""
    policy_id: str
    name: str
    description: str
    enabled: bool = True
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    # escalation_rules format: [{"level": EscalationLevel, "timeout_minutes": int, "targets": List[str]}]

@dataclass
class AlertMetrics:
    """Alert system metrics"""
    total_alerts: int = 0
    active_alerts: int = 0
    resolved_alerts: int = 0
    acknowledged_alerts: int = 0
    suppressed_alerts: int = 0
    notifications_sent: int = 0
    notification_failures: int = 0
    average_resolution_time: float = 0.0
    escalated_alerts: int = 0

class ConditionEvaluator:
    """Alert condition evaluation engine"""
    
    def __init__(self):
        self.operators = {
            '>': lambda x, y: float(x) > float(y),
            '<': lambda x, y: float(x) < float(y),
            '>=': lambda x, y: float(x) >= float(y),
            '<=': lambda x, y: float(x) <= float(y),
            '==': lambda x, y: str(x) == str(y),
            '!=': lambda x, y: str(x) != str(y),
            'contains': lambda x, y: str(y) in str(x),
            'not_contains': lambda x, y: str(y) not in str(x),
            'in': lambda x, y: str(x) in str(y).split(','),
            'not_in': lambda x, y: str(x) not in str(y).split(',')
        }
    
    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate alert condition against context"""
        try:
            # Simple condition parser
            # Format: "metric_name operator value"
            # Example: "cpu_usage > 80", "status == critical"
            
            parts = condition.strip().split()
            if len(parts) < 3:
                logger.error(f"Invalid condition format: {condition}")
                return False
            
            metric_name = parts[0]
            operator = parts[1]
            value = ' '.join(parts[2:])  # Handle values with spaces
            
            if metric_name not in context:
                logger.debug(f"Metric {metric_name} not found in context")
                return False
            
            if operator not in self.operators:
                logger.error(f"Unknown operator: {operator}")
                return False
            
            metric_value = context[metric_name]
            return self.operators[operator](metric_value, value)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def evaluate_complex_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate complex conditions with AND/OR logic"""
        try:
            # Handle AND/OR operators
            if ' AND ' in condition:
                sub_conditions = condition.split(' AND ')
                return all(self.evaluate_condition(cond.strip(), context) for cond in sub_conditions)
            elif ' OR ' in condition:
                sub_conditions = condition.split(' OR ')
                return any(self.evaluate_condition(cond.strip(), context) for cond in sub_conditions)
            else:
                return self.evaluate_condition(condition, context)
                
        except Exception as e:
            logger.error(f"Error evaluating complex condition '{condition}': {e}")
            return False

class NotificationDelivery:
    """Notification delivery system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.delivery_handlers = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.LOG: self._send_log,
            NotificationChannel.CONSOLE: self._send_console,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.SMS: self._send_sms
        }
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
    
    async def send_notification(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send notification to target"""
        try:
            # Check if target is enabled
            if not target.enabled:
                return False
            
            # Check severity filter
            if target.severity_filter and alert.severity not in target.severity_filter:
                return False
            
            # Check escalation level filter
            if target.escalation_levels and alert.escalation_level not in target.escalation_levels:
                return False
            
            # Check quiet hours
            if self._is_quiet_hours(target):
                logger.debug(f"Skipping notification to {target.target_id} due to quiet hours")
                return False
            
            # Check rate limiting
            if not self._check_rate_limit(target):
                logger.warning(f"Rate limit exceeded for target {target.target_id}")
                return False
            
            # Send notification
            handler = self.delivery_handlers.get(target.channel)
            if handler:
                success = await handler(alert, target)
                if success:
                    self._record_notification(target)
                return success
            else:
                logger.error(f"Unknown notification channel: {target.channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification to {target.target_id}: {e}")
            return False
    
    def _is_quiet_hours(self, target: NotificationTarget) -> bool:
        """Check if current time is within quiet hours"""
        if not target.quiet_hours:
            return False
        
        current_hour = datetime.now().hour
        start_hour, end_hour = target.quiet_hours
        
        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:  # Overnight quiet hours
            return current_hour >= start_hour or current_hour < end_hour
    
    def _check_rate_limit(self, target: NotificationTarget) -> bool:
        """Check if rate limit allows sending notification"""
        current_time = time.time()
        target_rates = self.rate_limits[target.target_id]
        
        # Remove old entries (older than 1 hour)
        while target_rates and current_time - target_rates[0] > 3600:
            target_rates.popleft()
        
        # Check if under rate limit
        return len(target_rates) < target.rate_limit
    
    def _record_notification(self, target: NotificationTarget):
        """Record notification for rate limiting"""
        self.rate_limits[target.target_id].append(time.time())
    
    async def _send_email(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send email notification"""
        try:
            # Email configuration from target config
            smtp_server = target.config.get('smtp_server', 'localhost')
            smtp_port = target.config.get('smtp_port', 587)
            username = target.config.get('username')
            password = target.config.get('password')
            use_tls = target.config.get('use_tls', True)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = target.config.get('from_address', 'cortexos@localhost')
            msg['To'] = target.address
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Timestamp: {alert.timestamp}
- Description: {alert.description}

Context:
{json.dumps(alert.context, indent=2)}

Status: {alert.status.value}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (simulated for testing)
            logger.info(f"EMAIL NOTIFICATION: {target.address} - {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp
            
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'context': alert.context,
                'tags': alert.tags
            }
            
            headers = target.config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    target.address,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status < 400:
                        logger.info(f"WEBHOOK NOTIFICATION: {target.address} - {alert.title}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_log(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send log notification"""
        try:
            log_level = target.config.get('log_level', 'INFO')
            log_message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}"
            
            if log_level.upper() == 'ERROR':
                logger.error(log_message)
            elif log_level.upper() == 'WARNING':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending log notification: {e}")
            return False
    
    async def _send_console(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send console notification"""
        try:
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.EMERGENCY: "🆘"
            }
            
            emoji = severity_emoji.get(alert.severity, "📢")
            print(f"{emoji} ALERT [{alert.severity.value.upper()}] {alert.title}")
            print(f"   Source: {alert.source}")
            print(f"   Time: {alert.timestamp}")
            print(f"   Description: {alert.description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending console notification: {e}")
            return False
    
    async def _send_slack(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send Slack notification"""
        try:
            # Slack webhook implementation (simulated)
            logger.info(f"SLACK NOTIFICATION: {target.address} - {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_sms(self, alert: Alert, target: NotificationTarget) -> bool:
        """Send SMS notification"""
        try:
            # SMS implementation (simulated)
            logger.info(f"SMS NOTIFICATION: {target.address} - {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False

class AlertManager:
    """Advanced alerting and notification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_targets = {}
        self.escalation_policies = {}
        self.metrics = AlertMetrics()
        
        # Components
        self.condition_evaluator = ConditionEvaluator()
        self.notification_delivery = NotificationDelivery(self.config.get('notification', {}))
        
        # Configuration
        self.evaluation_interval = self.config.get('evaluation_interval', 30)
        self.cleanup_interval = self.config.get('cleanup_interval', 3600)
        self.max_alert_age_hours = self.config.get('max_alert_age_hours', 168)  # 1 week
        self.enable_auto_resolution = self.config.get('enable_auto_resolution', True)
        self.enable_escalation = self.config.get('enable_escalation', True)
        
        # State
        self.running = False
        self.evaluation_task = None
        self.cleanup_task = None
        self.escalation_task = None
        
        logger.info("Alert Manager initialized")
    
    async def start(self):
        """Start alert management"""
        try:
            self.running = True
            
            # Start evaluation task
            self.evaluation_task = asyncio.create_task(self._evaluation_loop())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Start escalation task if enabled
            if self.enable_escalation:
                self.escalation_task = asyncio.create_task(self._escalation_loop())
            
            logger.info("Alert Manager started")
            
        except Exception as e:
            logger.error(f"Error starting Alert Manager: {e}")
            raise
    
    async def stop(self):
        """Stop alert management"""
        try:
            self.running = False
            
            # Cancel tasks
            tasks = [self.evaluation_task, self.cleanup_task]
            if self.escalation_task:
                tasks.append(self.escalation_task)
            
            for task in tasks:
                if task:
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
            
            logger.info("Alert Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Alert Manager: {e}")
    
    def register_alert_rule(self, rule: AlertRule):
        """Register alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Registered alert rule: {rule.rule_id}")
    
    def unregister_alert_rule(self, rule_id: str):
        """Unregister alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Unregistered alert rule: {rule_id}")
    
    def register_notification_target(self, target: NotificationTarget):
        """Register notification target"""
        self.notification_targets[target.target_id] = target
        logger.info(f"Registered notification target: {target.target_id}")
    
    def register_escalation_policy(self, policy: EscalationPolicy):
        """Register escalation policy"""
        self.escalation_policies[policy.policy_id] = policy
        logger.info(f"Registered escalation policy: {policy.policy_id}")
    
    async def evaluate_alerts(self, context: Dict[str, Any]):
        """Evaluate alert rules against current context"""
        try:
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Evaluate rule condition
                    triggered = self.condition_evaluator.evaluate_complex_condition(rule.condition, context)
                    
                    if triggered:
                        await self._handle_alert_triggered(rule, context)
                    else:
                        await self._handle_alert_resolved(rule_id)
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error in alert evaluation: {e}")
    
    async def _handle_alert_triggered(self, rule: AlertRule, context: Dict[str, Any]):
        """Handle triggered alert"""
        try:
            # Check if alert already exists and is in cooldown
            existing_alert = self._find_active_alert(rule.rule_id)
            
            if existing_alert:
                # Check cooldown
                time_since_last = (datetime.now() - existing_alert.timestamp).total_seconds()
                if time_since_last < rule.cooldown_minutes * 60:
                    return  # Still in cooldown
                
                # Update existing alert
                existing_alert.timestamp = datetime.now()
                existing_alert.context = context
            else:
                # Create new alert
                alert = Alert(
                    alert_id=self._generate_alert_id(rule.rule_id),
                    rule_id=rule.rule_id,
                    title=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    source=context.get('source', 'unknown'),
                    timestamp=datetime.now(),
                    context=context,
                    tags=rule.tags.copy()
                )
                
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self.metrics.total_alerts += 1
                self.metrics.active_alerts += 1
                
                # Send notifications
                await self._send_alert_notifications(alert)
                
                logger.info(f"Alert triggered: {alert.alert_id} - {alert.title}")
            
        except Exception as e:
            logger.error(f"Error handling triggered alert: {e}")
    
    async def _handle_alert_resolved(self, rule_id: str):
        """Handle resolved alert"""
        try:
            existing_alert = self._find_active_alert(rule_id)
            
            if existing_alert and self.enable_auto_resolution:
                rule = self.alert_rules.get(rule_id)
                if rule and rule.auto_resolve:
                    await self.resolve_alert(existing_alert.alert_id, "auto-resolved")
            
        except Exception as e:
            logger.error(f"Error handling resolved alert: {e}")
    
    def _find_active_alert(self, rule_id: str) -> Optional[Alert]:
        """Find active alert for rule"""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
    
    def _generate_alert_id(self, rule_id: str) -> str:
        """Generate unique alert ID"""
        timestamp = int(time.time() * 1000000)
        return f"{rule_id}_{timestamp}"
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for alert"""
        try:
            notification_count = 0
            
            for target in self.notification_targets.values():
                try:
                    success = await self.notification_delivery.send_notification(alert, target)
                    if success:
                        notification_count += 1
                        self.metrics.notifications_sent += 1
                    else:
                        self.metrics.notification_failures += 1
                        
                except Exception as e:
                    logger.error(f"Error sending notification to {target.target_id}: {e}")
                    self.metrics.notification_failures += 1
            
            alert.notification_count += notification_count
            alert.last_notification = datetime.now()
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_timestamp = datetime.now()
                
                self.metrics.acknowledged_alerts += 1
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_timestamp = datetime.now()
                
                # Calculate resolution time
                resolution_time = (alert.resolved_timestamp - alert.timestamp).total_seconds()
                
                # Update metrics
                self.metrics.active_alerts -= 1
                self.metrics.resolved_alerts += 1
                
                # Update average resolution time
                if self.metrics.resolved_alerts > 1:
                    self.metrics.average_resolution_time = (
                        (self.metrics.average_resolution_time * (self.metrics.resolved_alerts - 1) + resolution_time)
                        / self.metrics.resolved_alerts
                    )
                else:
                    self.metrics.average_resolution_time = resolution_time
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by} (resolution time: {resolution_time:.1f}s)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def suppress_alert(self, alert_id: str, suppressed_by: str) -> bool:
        """Suppress alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                
                self.metrics.suppressed_alerts += 1
                logger.info(f"Alert suppressed: {alert_id} by {suppressed_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error suppressing alert {alert_id}: {e}")
            return False
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop"""
        logger.info("Alert evaluation loop started")
        
        while self.running:
            try:
                # This would typically receive context from monitoring systems
                # For now, we'll use a placeholder context
                context = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'alert_manager'
                }
                
                await self.evaluate_alerts(context)
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Alert evaluation loop stopped")
    
    async def _cleanup_loop(self):
        """Alert cleanup loop"""
        logger.info("Alert cleanup loop started")
        
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in alert cleanup loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Alert cleanup loop stopped")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.max_alert_age_hours)
            
            # Clean up alert history
            while self.alert_history and self.alert_history[0].timestamp < cutoff_time:
                self.alert_history.popleft()
            
            # Auto-expire old active alerts
            expired_alerts = []
            for alert_id, alert in self.active_alerts.items():
                if alert.timestamp < cutoff_time:
                    expired_alerts.append(alert_id)
            
            for alert_id in expired_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.EXPIRED
                del self.active_alerts[alert_id]
                self.metrics.active_alerts -= 1
                logger.info(f"Alert expired: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def _escalation_loop(self):
        """Alert escalation loop"""
        logger.info("Alert escalation loop started")
        
        while self.running:
            try:
                await self._process_escalations()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in alert escalation loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Alert escalation loop stopped")
    
    async def _process_escalations(self):
        """Process alert escalations"""
        try:
            current_time = datetime.now()
            
            for alert in self.active_alerts.values():
                if alert.status != AlertStatus.ACTIVE:
                    continue
                
                rule = self.alert_rules.get(alert.rule_id)
                if not rule or not rule.escalation_enabled:
                    continue
                
                # Check if escalation timeout has passed
                time_since_alert = (current_time - alert.timestamp).total_seconds()
                
                if time_since_alert > rule.escalation_timeout and not alert.escalated_timestamp:
                    await self._escalate_alert(alert)
            
        except Exception as e:
            logger.error(f"Error processing escalations: {e}")
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert to next level"""
        try:
            # Simple escalation logic - move to next level
            current_level = alert.escalation_level
            
            if current_level == EscalationLevel.LEVEL_1:
                alert.escalation_level = EscalationLevel.LEVEL_2
            elif current_level == EscalationLevel.LEVEL_2:
                alert.escalation_level = EscalationLevel.LEVEL_3
            elif current_level == EscalationLevel.LEVEL_3:
                alert.escalation_level = EscalationLevel.EXECUTIVE
            
            alert.escalated_timestamp = datetime.now()
            self.metrics.escalated_alerts += 1
            
            # Send escalation notifications
            await self._send_alert_notifications(alert)
            
            logger.warning(f"Alert escalated: {alert.alert_id} to {alert.escalation_level.value}")
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert.alert_id}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.active_alerts.get(alert_id)
    
    def get_metrics(self) -> AlertMetrics:
        """Get alert system metrics"""
        # Update active alerts count
        self.metrics.active_alerts = len(self.active_alerts)
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status"""
        return {
            'running': self.running,
            'registered_rules': len(self.alert_rules),
            'notification_targets': len(self.notification_targets),
            'escalation_policies': len(self.escalation_policies),
            'active_alerts': len(self.active_alerts),
            'total_alerts_in_history': len(self.alert_history),
            'evaluation_interval': self.evaluation_interval,
            'enable_auto_resolution': self.enable_auto_resolution,
            'enable_escalation': self.enable_escalation,
            'metrics': {
                'total_alerts': self.metrics.total_alerts,
                'active_alerts': self.metrics.active_alerts,
                'resolved_alerts': self.metrics.resolved_alerts,
                'acknowledged_alerts': self.metrics.acknowledged_alerts,
                'notifications_sent': self.metrics.notifications_sent,
                'notification_failures': self.metrics.notification_failures,
                'average_resolution_time': self.metrics.average_resolution_time,
                'escalated_alerts': self.metrics.escalated_alerts
            }
        }

# Test and demonstration
async def test_alert_manager():
    """Test the alert manager system"""
    print("🧠 Testing CortexOS Alert Manager...")
    
    # Create configuration
    config = {
        'evaluation_interval': 5,  # 5 seconds for testing
        'cleanup_interval': 60,    # 1 minute for testing
        'max_alert_age_hours': 1,  # 1 hour for testing
        'enable_auto_resolution': True,
        'enable_escalation': True
    }
    
    # Initialize alert manager
    manager = AlertManager(config)
    
    # Register alert rules
    cpu_rule = AlertRule(
        rule_id="high_cpu",
        name="High CPU Usage",
        description="CPU usage is above 80%",
        condition="cpu_usage > 80",
        severity=AlertSeverity.WARNING,
        cooldown_minutes=5,
        escalation_enabled=True,
        escalation_timeout=30  # 30 seconds for testing
    )
    
    memory_rule = AlertRule(
        rule_id="high_memory",
        name="High Memory Usage",
        description="Memory usage is above 90%",
        condition="memory_usage > 90",
        severity=AlertSeverity.CRITICAL,
        cooldown_minutes=3
    )
    
    manager.register_alert_rule(cpu_rule)
    manager.register_alert_rule(memory_rule)
    
    # Register notification targets
    console_target = NotificationTarget(
        target_id="console",
        name="Console Notifications",
        channel=NotificationChannel.CONSOLE,
        address="console",
        severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
    )
    
    log_target = NotificationTarget(
        target_id="log",
        name="Log Notifications",
        channel=NotificationChannel.LOG,
        address="alert.log",
        config={'log_level': 'ERROR'}
    )
    
    manager.register_notification_target(console_target)
    manager.register_notification_target(log_target)
    
    try:
        # Start alert manager
        await manager.start()
        print("✅ Alert Manager started")
        
        # Simulate alert conditions
        print("\n📊 Simulating alert conditions...")
        
        # Trigger high CPU alert
        context1 = {
            'cpu_usage': 85.0,
            'memory_usage': 70.0,
            'source': 'system_monitor',
            'timestamp': datetime.now().isoformat()
        }
        
        await manager.evaluate_alerts(context1)
        await asyncio.sleep(2)
        
        # Trigger high memory alert
        context2 = {
            'cpu_usage': 75.0,
            'memory_usage': 95.0,
            'source': 'system_monitor',
            'timestamp': datetime.now().isoformat()
        }
        
        await manager.evaluate_alerts(context2)
        await asyncio.sleep(2)
        
        # Check active alerts
        active_alerts = manager.get_active_alerts()
        print(f"\n🚨 Active Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"   {alert.alert_id}: {alert.title} ({alert.severity.value}) - {alert.status.value}")
        
        # Acknowledge an alert
        if active_alerts:
            alert_to_ack = active_alerts[0]
            success = await manager.acknowledge_alert(alert_to_ack.alert_id, "test_user")
            print(f"\n✅ Alert acknowledged: {success}")
        
        # Wait for potential escalation
        print("\n⏳ Waiting for potential escalation...")
        await asyncio.sleep(35)
        
        # Resolve conditions
        context3 = {
            'cpu_usage': 60.0,
            'memory_usage': 70.0,
            'source': 'system_monitor',
            'timestamp': datetime.now().isoformat()
        }
        
        await manager.evaluate_alerts(context3)
        await asyncio.sleep(2)
        
        # Check alerts after resolution
        active_alerts_after = manager.get_active_alerts()
        print(f"\n📊 Active Alerts After Resolution: {len(active_alerts_after)}")
        
        # Display metrics
        print(f"\n📈 Alert Metrics:")
        metrics = manager.get_metrics()
        print(f"   Total Alerts: {metrics.total_alerts}")
        print(f"   Active Alerts: {metrics.active_alerts}")
        print(f"   Resolved Alerts: {metrics.resolved_alerts}")
        print(f"   Acknowledged Alerts: {metrics.acknowledged_alerts}")
        print(f"   Notifications Sent: {metrics.notifications_sent}")
        print(f"   Notification Failures: {metrics.notification_failures}")
        print(f"   Escalated Alerts: {metrics.escalated_alerts}")
        print(f"   Avg Resolution Time: {metrics.average_resolution_time:.1f}s")
        
        # Display alert history
        history = manager.get_alert_history(1)
        print(f"\n📚 Alert History (last hour): {len(history)} alerts")
        
        # Display manager status
        print(f"\n🔧 Manager Status:")
        status = manager.get_status()
        for key, value in status.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
        print("\n✅ Alert Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(test_alert_manager())

