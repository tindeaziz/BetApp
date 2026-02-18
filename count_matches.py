
from src.database import SupabaseDB
import time

try:
    db = SupabaseDB()
    # Supabase select count is tricky with filters, but plain select with head=True helps.
    # Or just select count(*).
    # Since Supabase python client doesn't support count() nicely directly on table object in all versions, 
    # we can try to select matching IDs with count='exact'.
    
    response = db.client.table('matches').select('match_id', count='exact').limit(1).execute()
    count = response.count
    print(f"Total Matches in DB: {count}")
    
except Exception as e:
    print(f"Error: {e}")
