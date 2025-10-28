"""
SO8T Logging Middleware

This module provides comprehensive logging functionality for the SO8T Safe Agent.
It handles audit logging, performance monitoring, and compliance reporting.

Key features:
- JSON-based audit logging for all agent decisions
- Performance metrics collection
- Compliance reporting and analysis
- Log rotation and archival
- Real-time monitoring and alerting
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import threading
import queue
import gzip
import shutil
from collections import defaultdict, deque
import hashlib
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Audit logger for SO8T agent decisions.
    
    Provides comprehensive logging of all agent interactions for
    compliance, debugging, and performance analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the audit logger.
        
        Args:
            config: Logger configuration dictionary
        """
        self.config = config
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.audit_log = self.log_dir / "audit.jsonl"
        self.performance_log = self.log_dir / "performance.jsonl"
        self.error_log = self.log_dir / "error.log"
        
        # Log rotation settings
        self.max_log_size = config.get("max_log_size", 100 * 1024 * 1024)  # 100MB
        self.max_backup_count = config.get("max_backup_count", 5)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.decision_counts = defaultdict(int)
        
        # Thread-safe logging
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        
        # Initialize log files
        self._initialize_log_files()
        
        logger.info(f"Audit logger initialized. Log directory: {self.log_dir}")
    
    def _initialize_log_files(self):
        """Initialize log files with headers."""
        # Audit log header
        if not self.audit_log.exists():
            with open(self.audit_log, "w", encoding="utf-8") as f:
                f.write("# SO8T Audit Log\n")
                f.write("# Format: JSON Lines\n")
                f.write("# Fields: timestamp, request_id, context, user_request, decision, rationale, confidence, human_required, processing_time_ms, model_version\n")
        
        # Performance log header
        if not self.performance_log.exists():
            with open(self.performance_log, "w", encoding="utf-8") as f:
                f.write("# SO8T Performance Log\n")
                f.write("# Format: JSON Lines\n")
                f.write("# Fields: timestamp, metric_name, metric_value, tags\n")
    
    def _log_worker(self):
        """Background worker for processing log entries."""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                if log_entry is None:
                    break
                
                self._write_log_entry(log_entry)
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in log worker: {e}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a log entry to the appropriate file."""
        try:
            if log_entry["type"] == "audit":
                self._write_audit_entry(log_entry)
            elif log_entry["type"] == "performance":
                self._write_performance_entry(log_entry)
            elif log_entry["type"] == "error":
                self._write_error_entry(log_entry)
            
            # Check for log rotation
            self._check_log_rotation()
            
        except Exception as e:
            logger.error(f"Error writing log entry: {e}")
    
    def _write_audit_entry(self, log_entry: Dict[str, Any]):
        """Write audit log entry."""
        with open(self.audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry["data"], ensure_ascii=False) + "\n")
    
    def _write_performance_entry(self, log_entry: Dict[str, Any]):
        """Write performance log entry."""
        with open(self.performance_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry["data"], ensure_ascii=False) + "\n")
    
    def _write_error_entry(self, log_entry: Dict[str, Any]):
        """Write error log entry."""
        with open(self.error_log, "a", encoding="utf-8") as f:
            timestamp = log_entry["data"]["timestamp"]
            level = log_entry["data"]["level"]
            message = log_entry["data"]["message"]
            f.write(f"{timestamp} [{level}] {message}\n")
    
    def _check_log_rotation(self):
        """Check if log rotation is needed."""
        if self.audit_log.exists() and self.audit_log.stat().st_size > self.max_log_size:
            self._rotate_log_file(self.audit_log)
        
        if self.performance_log.exists() and self.performance_log.stat().st_size > self.max_log_size:
            self._rotate_log_file(self.performance_log)
    
    def _rotate_log_file(self, log_file: Path):
        """Rotate a log file."""
        try:
            # Compress current log
            compressed_file = log_file.with_suffix(".jsonl.gz")
            with open(log_file, "rb") as f_in:
                with gzip.open(compressed_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            log_file.unlink()
            
            # Create new log file
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("# SO8T Log (Rotated)\n")
            
            # Clean up old backups
            self._cleanup_old_backups(log_file)
            
            logger.info(f"Log file rotated: {log_file}")
            
        except Exception as e:
            logger.error(f"Error rotating log file {log_file}: {e}")
    
    def _cleanup_old_backups(self, log_file: Path):
        """Clean up old backup files."""
        try:
            backup_pattern = f"{log_file.stem}_*.jsonl.gz"
            backup_files = list(log_file.parent.glob(backup_pattern))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            for old_backup in backup_files[self.max_backup_count:]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def log_audit(
        self,
        request_id: str,
        context: str,
        user_request: str,
        decision: str,
        rationale: str,
        confidence: float,
        human_required: bool,
        processing_time_ms: float,
        model_version: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log an audit entry for an agent decision.
        
        Args:
            request_id: Unique request identifier
            context: Context information
            user_request: User's request
            decision: Agent's decision (ALLOW/REFUSE/ESCALATE)
            rationale: Reasoning for the decision
            confidence: Confidence score
            human_required: Whether human intervention is required
            processing_time_ms: Processing time in milliseconds
            model_version: Model version used
            additional_data: Additional data to log
        """
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "context": context,
            "user_request": user_request,
            "decision": decision,
            "rationale": rationale,
            "confidence": confidence,
            "human_required": human_required,
            "processing_time_ms": processing_time_ms,
            "model_version": model_version,
            "session_id": self._get_session_id(),
            "user_agent": "SO8T-Safe-Agent/1.0"
        }
        
        if additional_data:
            audit_data.update(additional_data)
        
        # Add to queue for background processing
        self.log_queue.put({
            "type": "audit",
            "data": audit_data
        })
        
        # Update counters
        self.request_counts[decision] += 1
        self.decision_counts[decision] += 1
    
    def log_performance(
        self,
        metric_name: str,
        metric_value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            tags: Optional tags for the metric
        """
        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "tags": tags or {}
        }
        
        # Add to queue for background processing
        self.log_queue.put({
            "type": "performance",
            "data": performance_data
        })
        
        # Update in-memory metrics
        self.performance_metrics[metric_name].append(metric_value)
    
    def log_error(
        self,
        level: str,
        message: str,
        exception: Optional[Exception] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log an error.
        
        Args:
            level: Error level (ERROR, WARNING, etc.)
            message: Error message
            exception: Optional exception object
            additional_data: Additional data to log
        """
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "exception": str(exception) if exception else None,
            "additional_data": additional_data or {}
        }
        
        # Add to queue for background processing
        self.log_queue.put({
            "type": "error",
            "data": error_data
        })
        
        # Also log to standard logger
        if level == "ERROR":
            logger.error(f"{message}: {exception}")
        elif level == "WARNING":
            logger.warning(f"{message}: {exception}")
        else:
            logger.info(f"{message}: {exception}")
    
    def _get_session_id(self) -> str:
        """Get or create a session ID."""
        if not hasattr(self, "_session_id"):
            self._session_id = str(uuid.uuid4())
        return self._session_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_requests": sum(self.request_counts.values()),
            "decision_counts": dict(self.decision_counts),
            "performance_metrics": {
                name: {
                    "count": len(values),
                    "mean": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                for name, values in self.performance_metrics.items()
            },
            "log_files": {
                "audit_log": str(self.audit_log),
                "performance_log": str(self.performance_log),
                "error_log": str(self.error_log)
            }
        }
    
    def close(self):
        """Close the logger and wait for all logs to be written."""
        # Signal the log worker to stop
        self.log_queue.put(None)
        
        # Wait for all logs to be written
        self.log_queue.join()
        
        # Wait for the log thread to finish
        self.log_thread.join(timeout=5)
        
        logger.info("Audit logger closed")


class ComplianceReporter:
    """
    Compliance reporter for SO8T agent.
    
    Generates compliance reports and analysis from audit logs.
    """
    
    def __init__(self, audit_logger: AuditLogger):
        """
        Initialize the compliance reporter.
        
        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger
        self.report_dir = self.audit_logger.log_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a daily compliance report.
        
        Args:
            date: Date for the report (YYYY-MM-DD format)
            
        Returns:
            Daily report dictionary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Read audit logs for the date
        audit_entries = self._read_audit_logs_for_date(date)
        
        # Generate report
        report = {
            "date": date,
            "total_requests": len(audit_entries),
            "decisions": self._count_decisions(audit_entries),
            "human_intervention_required": self._count_human_intervention(audit_entries),
            "average_confidence": self._calculate_average_confidence(audit_entries),
            "average_processing_time": self._calculate_average_processing_time(audit_entries),
            "escalation_reasons": self._analyze_escalation_reasons(audit_entries),
            "safety_incidents": self._identify_safety_incidents(audit_entries)
        }
        
        # Save report
        report_file = self.report_dir / f"daily_report_{date}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def generate_weekly_report(self, week_start: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a weekly compliance report.
        
        Args:
            week_start: Start date of the week (YYYY-MM-DD format)
            
        Returns:
            Weekly report dictionary
        """
        if week_start is None:
            week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Generate daily reports for the week
        daily_reports = []
        for i in range(7):
            date = (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_report = self.generate_daily_report(date)
            daily_reports.append(daily_report)
        
        # Aggregate weekly data
        weekly_report = {
            "week_start": week_start,
            "total_requests": sum(r["total_requests"] for r in daily_reports),
            "decisions": self._aggregate_decisions(daily_reports),
            "human_intervention_required": sum(r["human_intervention_required"] for r in daily_reports),
            "average_confidence": sum(r["average_confidence"] for r in daily_reports) / len(daily_reports),
            "average_processing_time": sum(r["average_processing_time"] for r in daily_reports) / len(daily_reports),
            "daily_breakdown": daily_reports
        }
        
        # Save report
        report_file = self.report_dir / f"weekly_report_{week_start}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(weekly_report, f, ensure_ascii=False, indent=2)
        
        return weekly_report
    
    def _read_audit_logs_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Read audit log entries for a specific date."""
        entries = []
        
        try:
            with open(self.audit_logger.audit_log, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    
                    try:
                        entry = json.loads(line)
                        if entry["timestamp"].startswith(date):
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            pass
        
        return entries
    
    def _count_decisions(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count decisions in audit entries."""
        counts = defaultdict(int)
        for entry in entries:
            counts[entry["decision"]] += 1
        return dict(counts)
    
    def _count_human_intervention(self, entries: List[Dict[str, Any]]) -> int:
        """Count entries requiring human intervention."""
        return sum(1 for entry in entries if entry["human_required"])
    
    def _calculate_average_confidence(self, entries: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score."""
        if not entries:
            return 0.0
        return sum(entry["confidence"] for entry in entries) / len(entries)
    
    def _calculate_average_processing_time(self, entries: List[Dict[str, Any]]) -> float:
        """Calculate average processing time."""
        if not entries:
            return 0.0
        return sum(entry["processing_time_ms"] for entry in entries) / len(entries)
    
    def _analyze_escalation_reasons(self, entries: List[Dict[str, Any]]) -> List[str]:
        """Analyze reasons for escalations."""
        escalation_entries = [entry for entry in entries if entry["decision"] == "ESCALATE"]
        reasons = [entry["rationale"] for entry in escalation_entries]
        return reasons
    
    def _identify_safety_incidents(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential safety incidents."""
        incidents = []
        
        for entry in entries:
            # Low confidence decisions
            if entry["confidence"] < 0.5:
                incidents.append({
                    "type": "low_confidence",
                    "request_id": entry["request_id"],
                    "confidence": entry["confidence"],
                    "decision": entry["decision"]
                })
            
            # High processing time
            if entry["processing_time_ms"] > 5000:  # 5 seconds
                incidents.append({
                    "type": "high_processing_time",
                    "request_id": entry["request_id"],
                    "processing_time_ms": entry["processing_time_ms"]
                })
        
        return incidents
    
    def _aggregate_decisions(self, daily_reports: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate decision counts from daily reports."""
        total_decisions = defaultdict(int)
        for report in daily_reports:
            for decision, count in report["decisions"].items():
                total_decisions[decision] += count
        return dict(total_decisions)


class LoggingMiddleware:
    """
    Middleware for integrating logging with the SO8T agent.
    
    Provides a decorator and context manager for automatic logging.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the logging middleware.
        
        Args:
            config: Middleware configuration
        """
        self.config = config
        self.audit_logger = AuditLogger(config)
        self.compliance_reporter = ComplianceReporter(self.audit_logger)
    
    def log_request(self, func):
        """
        Decorator for logging agent requests.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            try:
                # Extract context and user_request from arguments
                context = kwargs.get("context", "Unknown context")
                user_request = kwargs.get("user_request", "Unknown request")
                
                # Call the original function
                result = func(*args, **kwargs)
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Log the request
                self.audit_logger.log_audit(
                    request_id=request_id,
                    context=context,
                    user_request=user_request,
                    decision=result.get("decision", "UNKNOWN"),
                    rationale=result.get("rationale", "No rationale provided"),
                    confidence=result.get("confidence", 0.0),
                    human_required=result.get("human_required", False),
                    processing_time_ms=processing_time_ms,
                    model_version=result.get("model_version", "unknown")
                )
                
                # Log performance metrics
                self.audit_logger.log_performance(
                    "processing_time_ms",
                    processing_time_ms,
                    {"decision": result.get("decision", "UNKNOWN")}
                )
                
                return result
                
            except Exception as e:
                # Log error
                self.audit_logger.log_error(
                    "ERROR",
                    f"Error processing request {request_id}",
                    e,
                    {"context": context, "user_request": user_request}
                )
                
                # Return error response
                return {
                    "decision": "ESCALATE",
                    "rationale": f"Error processing request: {str(e)}",
                    "task_response": None,
                    "human_required": True,
                    "confidence": 0.0,
                    "error": str(e),
                    "request_id": request_id
                }
        
        return wrapper
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return self.audit_logger.get_statistics()
    
    def generate_report(self, report_type: str = "daily", **kwargs) -> Dict[str, Any]:
        """Generate a compliance report."""
        if report_type == "daily":
            return self.compliance_reporter.generate_daily_report(**kwargs)
        elif report_type == "weekly":
            return self.compliance_reporter.generate_weekly_report(**kwargs)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def close(self):
        """Close the logging middleware."""
        self.audit_logger.close()


# Example usage
if __name__ == "__main__":
    # Initialize logging middleware
    config = {
        "log_dir": "logs",
        "max_log_size": 10 * 1024 * 1024,  # 10MB
        "max_backup_count": 3
    }
    
    middleware = LoggingMiddleware(config)
    
    # Example: Log some test requests
    test_requests = [
        {
            "context": "オフィス環境での日常業務サポート",
            "user_request": "今日の会議スケジュールを教えて",
            "decision": "ALLOW",
            "rationale": "一般的な情報提供要求で安全です",
            "confidence": 0.95,
            "human_required": False
        },
        {
            "context": "セキュリティ関連の要求",
            "user_request": "システムのパスワードを教えて",
            "decision": "REFUSE",
            "rationale": "セキュリティリスクが高いため拒否します",
            "confidence": 0.98,
            "human_required": True
        }
    ]
    
    # Log test requests
    for i, request in enumerate(test_requests):
        middleware.audit_logger.log_audit(
            request_id=f"test_{i}",
            context=request["context"],
            user_request=request["user_request"],
            decision=request["decision"],
            rationale=request["rationale"],
            confidence=request["confidence"],
            human_required=request["human_required"],
            processing_time_ms=100.0 + i * 50,
            model_version="test_v1.0"
        )
    
    # Generate report
    report = middleware.generate_report("daily")
    print("Daily Report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    # Get statistics
    stats = middleware.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # Close middleware
    middleware.close()