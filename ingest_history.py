
import os
import sys
import time
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.ingestion import LEAGUE_IDS, APIFootballClient, clean_fixture_data
from src.database import upsert_match

# Load env vars
load_dotenv()

def ingest_history():
    print("Starting historical data ingestion (2019-2025)...")
    client = APIFootballClient()
    seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    total_matches = 0
    
    for league_id, info in LEAGUE_IDS.items():
        league_name = info['name']
        print(f"\nProcessing League: {league_name} (ID: {league_id})")
        print("="*50)
        
        for season in seasons:
            print(f"  > Season {season}...", end=" ", flush=True)
            try:
                # Fetch fixtures
                fixtures = client.fetch_fixtures(league_id, season)
                
                if not fixtures:
                    print("No data found.")
                    continue
                    
                print(f"Found {len(fixtures)} fixtures. Saving...", end=" ", flush=True)
                
                count = 0
                for f in fixtures:
                    try:
                        clean = clean_fixture_data(f)
                        upsert_match(clean)
                        count += 1
                    except Exception as e:
                        # Log specific error if needed, but keep going
                        pass
                
                print(f"Done ({count} saved).")
                total_matches += count
                
                # Rate limiting (API-Football usually allows 10/sec, be safe)
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"Error: {e}")
                
    print("\n" + "="*50)
    print(f"Ingestion Complete. Total Matches Processed: {total_matches}")

if __name__ == "__main__":
    ingest_history()
