import time
import pytest
from backtester import _attempt_call_existing_simulator

def make_signal(identifier, entry, resolved=True, source_job_id="job1"):
    return {
        "resolved": resolved,
        "identifier": identifier,
        "resolved_price": entry,
        "source_job_id": source_job_id
    }

@pytest.mark.asyncio
async def test_dedup_and_cooldown_behavior():
    # three signals for same identifier within dedup window -> only one should be simulated
    s1 = make_signal("FOO", 10)
    s2 = make_signal("FOO", 11)
    s3 = make_signal("BAR", 5)  # different symbol, should run
    config = {"dedup_window_seconds": 60, "cooldown_seconds_per_asset": 0}
    result = await _attempt_call_existing_simulator([s1, s2, s3], config)
    assert result["summary"]["closed"] == 2  # FOO deduped to 1, BAR runs

    # cooldown per asset: simulate two separated signals for same id, cooldown prevents second
    s4 = make_signal("BAZ", 20, source_job_id="jobA")
    s5 = make_signal("BAZ", 21, source_job_id="jobA")
    config2 = {"dedup_window_seconds": 0, "cooldown_seconds_per_asset": 60}
    res2 = await _attempt_call_existing_simulator([s4, s5], config2)
    assert res2["summary"]["closed"] == 1
