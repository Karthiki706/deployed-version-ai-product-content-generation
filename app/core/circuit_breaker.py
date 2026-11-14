import time
from threading import Lock
from app.core.logger import logger

class CircuitBreaker:
    """Independent Circuit Breaker instance per model name (Multiton pattern)."""
    _instances = {}
    _lock = Lock()

    def __new__(cls, model_name: str, *args, **kwargs):
        with cls._lock:
            if model_name not in cls._instances:
                instance = super(CircuitBreaker, cls).__new__(cls)
                instance._initialized = False
                cls._instances[model_name] = instance
            return cls._instances[model_name]

    def __init__(self, model_name: str, failure_threshold: int = 3, recovery_time: float = 60.0):
        if self._initialized:
            return
        self.model_name = model_name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.consecutive_failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_state_change = time.time()
        self.lock = Lock()
        self._initialized = True

    def can_execute(self) -> bool:
        """
        Evaluates circuit health for this specific model.
        
        Returns:
            bool: True if calls can proceed (CLOSED/HALF_OPEN), False if blocked (OPEN).
        """
        with self.lock:
            now = time.time()
            if self.state == "OPEN":
                elapsed = now - self.last_state_change
                if elapsed >= self.recovery_time:
                    logger.info(f"Circuit Breaker cooldown period elapsed for model '{self.model_name}'. Transitioning to HALF_OPEN.")
                    self.state = "HALF_OPEN"
                    self.last_state_change = now
                    return True
                return False
            return True

    def record_success(self):
        """Resets the circuit back to CLOSED state on a successful request."""
        with self.lock:
            if self.state != "CLOSED":
                logger.info(f"API success recorded. Resetting circuit for '{self.model_name}' from {self.state} to CLOSED.")
            self.state = "CLOSED"
            self.consecutive_failures = 0
            self.last_state_change = time.time()

    def record_failure(self):
        """Increments failures and trips/returns the circuit to OPEN state."""
        with self.lock:
            self.consecutive_failures += 1
            now = time.time()
            logger.warning(f"Recorded API failure for '{self.model_name}' ({self.consecutive_failures}/{self.failure_threshold}).")
            
            if self.state == "HALF_OPEN":
                logger.error(f"Probe request failed for '{self.model_name}' in HALF_OPEN state. Transitioning back to OPEN and restarting cooldown.")
                self.state = "OPEN"
                self.last_state_change = now
            elif self.consecutive_failures >= self.failure_threshold:
                if self.state != "OPEN":
                    logger.error(f"Consecutive failures exceeded limit ({self.failure_threshold}) for '{self.model_name}'. Tripping circuit to OPEN for {self.recovery_time}s.")
                    self.state = "OPEN"
                    self.last_state_change = now
