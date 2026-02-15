# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Market hours and trading calendar utilities."""

from datetime import datetime, time, date, timedelta
from typing import Optional, List
import pytz
import requests
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger()


class MarketHours:
    """
    Manage market hours and trading calendar.
    
    Determines when real-time streaming should be active based on:
    - Trading days (Monday-Friday)
    - Market hours (9:30 AM - 4:00 PM ET)
    - Market holidays (from Polygon API)
    - Pre-market/after-hours: DISABLED
    """
    
    def __init__(self, config: dict, api_key: str):
        """
        Initialize market hours manager.
        
        Args:
            config: Configuration dict with market_hours settings
            api_key: Polygon API key for fetching market calendar
        """
        self.api_key = api_key
        self.timezone = pytz.timezone(config.get("timezone", "America/New_York"))
        self.active_days = config.get("active_days", [0, 1, 2, 3, 4])  # Mon-Fri
        self.start_time = self._parse_time(config.get("start_time", "09:30"))
        self.end_time = self._parse_time(config.get("end_time", "16:00"))
        
        # Pre-market and after-hours DISABLED
        self.include_premarket = False
        self.include_afterhours = False
        
        # Market holidays fetched from Polygon API
        self.holidays: List[date] = []
        self._load_market_holidays()
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            check_time: Time to check (default: now)
            
        Returns:
            True if market is open, False otherwise
        """
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # Check if weekend
        if check_time.weekday() not in self.active_days:
            return False
        
        # Check if holiday
        if self._is_holiday(check_time.date()):
            return False
        
        # Check time range (regular hours only: 9:30 AM - 4:00 PM ET)
        current_time = check_time.time()
        if self.start_time <= current_time <= self.end_time:
            return True
        
        return False
    
    def seconds_until_market_open(self) -> Optional[int]:
        """
        Calculate seconds until next market open.
        
        Returns:
            Seconds until market opens, or None if currently open
        """
        now = datetime.now(self.timezone)
        
        if self.is_market_open(now):
            return None
        
        # Find next market day
        next_open = now.replace(
            hour=self.start_time.hour,
            minute=self.start_time.minute,
            second=0,
            microsecond=0
        )
        
        # If past today's close, move to next day
        if now.time() > self.end_time:
            next_open += timedelta(days=1)
        
        # Skip weekends and holidays
        while next_open.weekday() not in self.active_days or self._is_holiday(next_open.date()):
            next_open += timedelta(days=1)
        
        return int((next_open - now).total_seconds())
    
    def seconds_until_market_close(self) -> Optional[int]:
        """
        Calculate seconds until market close.
        
        Returns:
            Seconds until market closes, or None if currently closed
        """
        now = datetime.now(self.timezone)
        
        if not self.is_market_open(now):
            return None
        
        market_close = now.replace(
            hour=self.end_time.hour,
            minute=self.end_time.minute,
            second=0,
            microsecond=0
        )
        
        return int((market_close - now).total_seconds())
    
    @lru_cache(maxsize=1)
    def _load_market_holidays(self) -> None:
        """
        Load market holidays from Polygon API.
        
        Fetches current year holidays and caches the result.
        Falls back to empty list if API call fails.
        """
        try:
            current_year = datetime.now().year
            url = f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract holiday dates from response
            # Polygon returns market status including holidays
            if isinstance(data, list):
                for item in data:
                    if item.get("status") == "closed" and "date" in item:
                        holiday_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
                        self.holidays.append(holiday_date)
            
            # Sort holidays
            self.holidays.sort()
            
        except Exception as e:
            # Log error but don't fail - use empty holiday list as fallback
            logger.warning(f"Failed to load market holidays from API: {e}. Continuing with empty holiday list")
            self.holidays = []
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string (HH:MM) to time object."""
        hour, minute = map(int, time_str.split(":"))
        return time(hour, minute)
    
    def _is_holiday(self, check_date: date) -> bool:
        """Check if date is a market holiday."""
        return check_date in self.holidays

