from __future__ import annotations
from dataclasses import dataclass, field
import threading

@dataclass
class PipelineStats:
    llm_calls: int = 0
    cached_llm_calls: int = 0
    failed_llm_calls: int = 0
    fallback_calls: int = 0
    
    total_rows: int = 0
    processed_rows: int = 0
    
    current_stage_total: int = 0
    current_stage_processed: int = 0
    
    row_rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_row_rejection(self, reason: str):
        with self._lock:
            self.row_rejection_reason_counts[reason] = self.row_rejection_reason_counts.get(reason, 0) + 1

    def record_llm_call(self, cached: bool = False, failed: bool = False, fallback: bool = False):
        with self._lock:
            if cached:
                self.cached_llm_calls += 1
            elif failed:
                self.failed_llm_calls += 1
            else:
                self.llm_calls += 1
            
            if fallback:
                self.fallback_calls += 1

    def record_row_processed(self, increment: int = 1):
        with self._lock:
            self.current_stage_processed += increment

    def reset_stage(self, total: int):
        with self._lock:
            self.current_stage_total = total
            self.current_stage_processed = 0

# Global singleton for current run
GLOBAL_STATS = PipelineStats()
