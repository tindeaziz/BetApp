
from src.database import SupabaseDB
import pandas as pd

def debug_join():
    db = SupabaseDB()
    
    # Get 1 match
    matches = db.client.table('matches').select('*').limit(1).execute()
    if not matches.data:
        print("No matches found.")
        return
        
    m = matches.data[0]
    mid = m['match_id']
    h_team = m['home_team']
    a_team = m['away_team']
    
    print(f"Match ID: {mid}")
    print(f"Home Team (Matches): '{h_team}'")
    print(f"Away Team (Matches): '{a_team}'")
    
    # Get stats for this match
    stats = db.client.table('stats').select('*').eq('match_id', mid).execute()
    
    if not stats.data:
        print("No stats found for this match.")
    else:
        print(f"Stats found: {len(stats.data)} rows.")
        for s in stats.data:
            print(f"Stats Team: '{s.get('team')}' (ID: {s.get('team_id')}) - Fouls: {s.get('fouls')}")
            
if __name__ == "__main__":
    debug_join()
