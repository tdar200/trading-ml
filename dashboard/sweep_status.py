"""Display-only sweep progress. Failed downloads never count as backtests."""
from datetime import datetime, timezone
import fcntl


def snapshot(root, status, now=None):
    states=status.get('states',{})
    total=sum(states.values())
    pending=states.get('PENDING',0)
    tested=states.get('GPU_TESTED',0)
    attempted=total-pending
    active=False
    try:
        with (root/'worker.lock').open('r') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
                fcntl.flock(lock,fcntl.LOCK_UN)
            except BlockingIOError:
                active=True
    except FileNotFoundError:
        pass
    reason=(root/'STOP').read_text().strip() if (root/'STOP').exists() else ''
    now=now or datetime.now(timezone.utc)
    try:age=max(0,(now-datetime.fromisoformat(status['updated_at'])).total_seconds())
    except (KeyError,ValueError,TypeError):age=None
    if reason or (root/'STOP').exists():
        state='Stopping after current batch' if active else 'Paused'
    elif active:
        state='Running — update overdue' if age is not None and age>180 else 'Running'
    elif status.get('stage')=='AVAILABLE_DATA_SWEEP_COMPLETE':state='Finished available queue'
    else:state='Worker not running'
    return dict(total=total,attempted=attempted,pending=pending,tested=tested,
        ready=states.get('READY',0),excluded=states.get('EXCLUDED',0),errors=states.get('FETCH_ERROR',0),
        attempted_fraction=attempted/total if total else 0,tested_fraction=tested/total if total else 0,
        active=active,state=state,reason=reason,age_seconds=age)
