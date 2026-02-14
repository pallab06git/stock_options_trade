"""Abstract base class for all data sinks."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum


class SinkType(Enum):
    """Types of data sinks."""
    PARQUET = "parquet"
    SQL = "sql"
    CLOUD = "cloud"


class BaseSink(ABC):
    """
    Abstract base class for data sink implementations.
    
    All sinks (Parquet, SQL, Cloud) must implement this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sink with configuration.
        
        Args:
            config: Configuration dictionary containing paths, credentials, etc.
        """
        self.config = config
        self.sink_type: Optional[SinkType] = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to sink (if applicable)."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to sink."""
        pass
    
    @abstractmethod
    def write_batch(
        self,
        records: List[Dict[str, Any]],
        partition_key: Optional[str] = None
    ) -> None:
        """
        Write batch of records to sink.
        
        Args:
            records: List of data records
            partition_key: Optional partition identifier (e.g., date)
        """
        pass
    
    @abstractmethod
    def write_single(
        self,
        record: Dict[str, Any],
        partition_key: Optional[str] = None
    ) -> None:
        """
        Write single record to sink.
        
        Args:
            record: Single data record
            partition_key: Optional partition identifier
        """
        pass
    
    @abstractmethod
    def check_duplicate(self, record: Dict[str, Any]) -> bool:
        """
        Check if record already exists in sink.
        
        Args:
            record: Data record to check
            
        Returns:
            True if duplicate exists, False otherwise
        """
        pass
    
    @abstractmethod
    def overwrite(
        self,
        records: List[Dict[str, Any]],
        partition_key: str
    ) -> None:
        """
        Overwrite existing data in partition.
        
        Args:
            records: New records to write
            partition_key: Partition to overwrite
        """
        pass
