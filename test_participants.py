#!/usr/bin/env python3
"""
Test participant data availability
"""

import sys
sys.path.append('.')
from indoor_electrode_analysis import get_participants_with_both_sessions, get_available_participants
from pathlib import Path

print('🔍 Testing participant filtering...')
print('=' * 50)

all_participants = get_available_participants()
both_participants = get_participants_with_both_sessions()

print(f'📋 All participants found: {len(all_participants)}')
print(f'    {all_participants}')
print(f'📋 Participants with both sessions: {len(both_participants)}')
print(f'    {both_participants}')

print('\n📊 Individual participant status:')
processed_dir = Path('results/processed')
for participant in all_participants:
    participant_dir = processed_dir / participant
    has_indoor = (participant_dir / 'indoor_processed-epo.fif').exists()
    has_outdoor = (participant_dir / 'outdoor_processed-epo.fif').exists()
    status = []
    if has_indoor: 
        status.append('indoor')
    if has_outdoor: 
        status.append('outdoor')
    print(f'  {participant}: {status}')

print('\n✅ Test completed successfully!')
