"""
Script to fix timestamps in the database by re-fetching from Portkey.

The original ingestion used datetime.fromisoformat() which couldn't parse
Portkey's JavaScript Date format: "Sat Jan 17 2026 06:20:43 GMT+0000 (...)"

This script:
1. Fetches logs from Portkey with the correct timestamps
2. Updates the database records with the correct timestamps
"""
import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.models.log_entry import LogEntry
from app.services.ingestion.portkey_client import PortkeyClient
from app.services.ingestion.log_ingestion import parse_portkey_timestamp


async def fix_timestamps(dry_run: bool = True):
    """Fix timestamps in the database using correct parsing."""
    
    client = PortkeyClient()
    
    # Fetch logs from Portkey for the last week
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    print(f"Fetching logs from Portkey ({start_date.date()} to {end_date.date()})...")
    
    result = await client.get_logs(
        start_date=start_date,
        end_date=end_date,
        limit=1000
    )
    
    logs = result.get("logs", [])
    print(f"Fetched {len(logs)} logs from Portkey\n")
    
    # Create a mapping of Portkey log ID to correct timestamp
    portkey_timestamps = {}
    for log in logs:
        log_id = log.get("id")
        time_str = log.get("time_of_generation") or log.get("created_at")
        if log_id and time_str:
            correct_ts = parse_portkey_timestamp(time_str)
            portkey_timestamps[log_id] = correct_ts
    
    print(f"Parsed {len(portkey_timestamps)} timestamps\n")
    
    # Show sample of correct timestamps
    print("Sample of correct timestamps:")
    for log_id, ts in list(portkey_timestamps.items())[:3]:
        print(f"  {log_id}: {ts.isoformat()}")
    print()
    
    # Update database
    async with async_session_factory() as session:
        # Get all log entries
        result = await session.execute(select(LogEntry))
        db_logs = result.scalars().all()
        
        print(f"Found {len(db_logs)} logs in database\n")
        
        updated_count = 0
        skipped_count = 0
        not_found_count = 0
        
        for db_log in db_logs:
            portkey_id = db_log.portkey_log_id
            
            if portkey_id in portkey_timestamps:
                correct_ts = portkey_timestamps[portkey_id]
                old_ts = db_log.timestamp
                
                # Check if timestamp needs updating (more than 1 minute difference)
                if old_ts:
                    diff = abs((correct_ts - old_ts).total_seconds())
                    if diff < 60:
                        skipped_count += 1
                        continue
                
                if dry_run:
                    print(f"Would update {portkey_id}:")
                    print(f"  Old: {old_ts.isoformat() if old_ts else 'None'}")
                    print(f"  New: {correct_ts.isoformat()}")
                else:
                    db_log.timestamp = correct_ts
                updated_count += 1
            else:
                not_found_count += 1
        
        if not dry_run:
            await session.commit()
            print(f"\nCommitted {updated_count} updates to database")
        else:
            print(f"\n[DRY RUN] Would update {updated_count} logs")
        
        print(f"Skipped (already correct): {skipped_count}")
        print(f"Not found in Portkey response: {not_found_count}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix log timestamps in database")
    parser.add_argument("--fix", action="store_true", help="Actually apply fixes (default is dry-run)")
    args = parser.parse_args()
    
    dry_run = not args.fix
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("Run with --fix to apply changes")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("FIXING TIMESTAMPS - Changes will be committed")
        print("=" * 60)
        print()
    
    await fix_timestamps(dry_run=dry_run)


if __name__ == "__main__":
    asyncio.run(main())
