"""Run the actual public compile CLI under the established host-memory guard."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--config')
    group.add_argument('--command', nargs=argparse.REMAINDER)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--reserve-gib', type=float, default=4)
    a = p.parse_args()
    import psutil
    assert a.reserve_gib >= 4, 'Keep at least4GiB available host RAM'
    assert not a.report.exists()
    if psutil.virtual_memory().available < 24 * 1024**3:
        raise RuntimeError('Compile requires24GiB available host RAM')
    command = a.command if a.command is not None else [
        sys.executable, '-B', '-u', '-m', 'fluxhermes.cli.main', 'compile', a.config,
        '--execute'
    ]
    assert command
    report = dict(
        command=command,
        reserve_gib=a.reserve_gib,
        monitor_interval_s=0.1,
        max_rss_gib=0,
        min_available_gib=psutil.virtual_memory().available / 1024**3,
        aborted=None)
    child = subprocess.Popen(command, start_new_session=True)
    process = psutil.Process(child.pid)
    start = time.monotonic()
    pgid = child.pid

    def terminate_group(sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass

    def group_exists():
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def cleanup_group():
        if not group_exists():
            return
        terminate_group(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while group_exists():
            child.poll()
            if time.monotonic() >= deadline:
                terminate_group(signal.SIGKILL)
                report['cleanup_escalated'] = True
                break
            time.sleep(0.1)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGHUP, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while child.poll() is None:
            try:
                members = [process] + process.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss = 0
            for member in members:
                try:
                    rss += member.memory_info().rss / 1024**3
                except psutil.NoSuchProcess:
                    pass
            available = psutil.virtual_memory().available / 1024**3
            report['max_rss_gib'] = max(report['max_rss_gib'], rss)
            report['min_available_gib'] = min(report['min_available_gib'], available)
            report['elapsed_seconds'] = time.monotonic() - start
            a.report.write_text(json.dumps(report, indent=2) + '\n')
            if rss > 40 or available < a.reserve_gib:
                report['aborted'] = dict(
                    reason='host-memory guard', rss_gib=rss, available_gib=available)
                break
            try:
                child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        report['aborted'] = dict(reason='controller interrupted')
        raise
    finally:
        cleanup_group()
        report['exit_code'] = child.wait()
        report['elapsed_seconds'] = time.monotonic() - start
        a.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['exit_code'] == 0 and report['aborted'] is None else 1)


if __name__ == "__main__":
    main()
