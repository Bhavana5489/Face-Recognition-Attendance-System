import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON for easy ingestion by Prometheus/ELK.
    Implements Phase 17 Observability requirements.
    """
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        
        # Add any extra attributes passed to the logger
        if hasattr(record, 'metadata'):
            log_obj["metadata"] = record.metadata
            
        return json.dumps(log_obj)

def setup_logger(name: str = "gazepass") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    return logger

# Singleton export
logger = setup_logger()
