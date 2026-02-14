"""Heartbeat monitoring for real-time streaming."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import psutil


class HeartbeatMonitor:
    """
    Monitor real-time stream health via periodic heartbeats.
    
    Updates metadata every 5 minutes instead of logging every message.
    Alerts if no heartbeat for 15 minutes.
    """
    
    def __init__(self, config: dict, source_name: str):
        """
        Initialize heartbeat monitor.
        
        Args:
            config: Configuration dict with monitoring settings
            source_name: Name of data source (e.g., "polygon_spy")
        """
        self.source_name = source_name
        self.heartbeat_interval = config.get("heartbeat_interval_seconds", 300)  # 5 min
        self.alert_threshold = config.get("alert_threshold_seconds", 900)  # 15 min
        
        self.heartbeat_dir = Path("data/logs/heartbeat")
        self.heartbeat_file = self.heartbeat_dir / f"{source_name}_status.json"
        
        # Counters
        self.messages_since_last_heartbeat = 0
        self.total_messages_today = 0
        self.last_heartbeat_time: Optional[datetime] = None
        self.last_message_time: Optional[datetime] = None
    
    def record_message(self) -> None:
        """
        Record that a message was received.
        
        This is called for each WebSocket message but does NOT log it.
        Only increments counters for next heartbeat.
        """
        self.messages_since_last_heartbeat += 1
        self.total_messages_today += 1
        self.last_message_time = datetime.utcnow()
    
    def should_send_heartbeat(self) -> bool:
        """
        Check if it's time to send heartbeat.
        
        Returns:
            True if heartbeat interval has elapsed
        """
        if self.last_heartbeat_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_heartbeat_time).total_seconds()
        return elapsed >= self.heartbeat_interval
    
    def send_heartbeat(self) -> None:
        """
        Write heartbeat metadata to file.
        
        This is the ONLY logging done during streaming (every 5 min).
        """
        metadata = {
            "last_heartbeat": datetime.utcnow().isoformat() + "Z",
            "messages_received_last_5min": self.messages_since_last_heartbeat,
            "total_messages_today": self.total_messages_today,
            "last_message_time": self.last_message_time.isoformat() + "Z" if self.last_message_time else None,
            "connection_status": "active",
            "source": self.source_name,
            "memory_usage_mb": self._get_memory_usage()
        }
        
        # Write to file
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        with open(self.heartbeat_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Reset counter
        self.messages_since_last_heartbeat = 0
        self.last_heartbeat_time = datetime.utcnow()
    
    def check_stalled_stream(self) -> bool:
        """
        Check if stream has stalled (no heartbeat for 15 min).
        
        Returns:
            True if stream appears stalled and should alert
        """
        if not self.heartbeat_file.exists():
            return False
        
        with open(self.heartbeat_file) as f:
            metadata = json.load(f)
        
        last_heartbeat = datetime.fromisoformat(metadata["last_heartbeat"].rstrip("Z"))
        elapsed = (datetime.utcnow() - last_heartbeat).total_seconds()
        
        return elapsed > self.alert_threshold
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
