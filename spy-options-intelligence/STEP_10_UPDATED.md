# Updated Step 10: SPY Real-time Streaming (WebSocket)

## Implementation Requirements

```
Implement real-time SPY ingestion with resource-efficient streaming:

1. Complete src/utils/market_hours.py
   - Implement is_market_open() (9:30 AM - 4:00 PM ET, Mon-Fri)
   - Implement seconds_until_market_open()
   - Implement seconds_until_market_close()
   - Load market holidays from Polygon API: GET /v1/marketstatus/upcoming
   - Cache holidays for current year using @lru_cache
   - Fallback to empty list if API fails (log warning, don't fail)
   - Pre-market and after-hours: DISABLED (hardcoded False)

2. Complete src/monitoring/heartbeat_monitor.py
   - Implement record_message() (counter only, NO logging)
   - Implement send_heartbeat() (write metadata every 5 min)
   - Implement check_stalled_stream() (alert if >15 min)
   - Track: message count, memory usage, connection status

3. Update src/data_sources/polygon_client.py
   - Add stream_realtime() method
   - Use WebSocketClient from polygon library
   - Subscribe to "A.SPY"
   - Check market hours BEFORE connecting
   - Handle reconnection on disconnect
   - Call heartbeat.record_message() for each message (NO direct logging)
   - Call heartbeat.send_heartbeat() every 5 minutes
   - Buffer messages before writing to sink
   - Graceful shutdown at market close

4. Create src/orchestrator/streaming_runner.py
   - Initialize WebSocket client
   - Check market hours before starting
   - Initialize HeartbeatMonitor
   - Continuous write to Parquet (micro-batches every 30 sec)
   - Monitor heartbeat, alert if stalled
   - Auto-shutdown outside market hours
   - Graceful shutdown on SIGTERM/SIGINT
   - Late data handling (reject older than current day)

5. Update config/settings.yaml
   - Add streaming.market_hours section:
     * timezone: "America/New_York"
     * active_days: [0, 1, 2, 3, 4]  # Mon-Fri
     * start_time: "09:30"
     * end_time: "16:00"
     * # No pre-market or after-hours config (disabled)
   - Add streaming.monitoring section:
     * heartbeat_interval_seconds: 300  # 5 minutes
     * alert_threshold_seconds: 900     # 15 minutes
   - Add streaming.batching:
     * write_interval_seconds: 30  # Write every 30 sec

CRITICAL - Logging Rules:
- DO NOT log every WebSocket message
- DO NOT write individual ticks to execution logs
- ONLY log:
  * Stream start/stop events
  * Heartbeat updates (every 5 min to heartbeat file)
  * Errors/alerts
  * Market hours status changes

Test: 
1. Start stream during off-market hours → Should refuse to start
2. Start stream during market hours → Should connect
3. Verify heartbeat file updates every 5 minutes
4. Simulate 20-min stream stall → Should trigger alert
5. Verify auto-shutdown at 4:00 PM ET
6. Check log file sizes are minimal (no per-message logging)

Stop after completion. Generate Task Completion Report.
```

## Why These Changes

**Problem:** Logging every WebSocket message creates:
- Multi-GB log files per day
- Disk I/O bottleneck
- Makes logs unsearchable

**Solution:** Heartbeat monitoring
- One metadata file per source
- Update every 5 minutes (not every message)
- Easy to monitor: just check heartbeat file
- Alerts if stream fails (>15 min no update)

**Resource Savings:**
- No compute during nights/weekends (market closed)
- Reduced log storage (MB instead of GB)
- Lower network usage (no unnecessary connections)
