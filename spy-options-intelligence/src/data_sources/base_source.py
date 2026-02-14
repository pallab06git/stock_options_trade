"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes for data sources."""
    HISTORICAL = "historical"
    REALTIME = "realtime"


class BaseSource(ABC):
    """
    Abstract base class for data source implementations.
    
    All data sources (Polygon, Massive, News) must implement this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data source with configuration.
        
        Args:
            config: Configuration dictionary containing API keys, endpoints, etc.
        """
        self.config = config
        self.mode: Optional[ExecutionMode] = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch historical data for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional source-specific parameters
            
        Yields:
            Data records as dictionaries
        """
        pass
    
    @abstractmethod
    def stream_realtime(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Stream real-time data via WebSocket.
        
        Args:
            **kwargs: Source-specific subscription parameters
            
        Yields:
            Data records as dictionaries
        """
        pass
    
    @abstractmethod
    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a single data record.
        
        Args:
            record: Data record to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
