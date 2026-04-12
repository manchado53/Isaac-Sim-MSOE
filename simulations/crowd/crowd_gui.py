"""Read omni.anim.people source to understand the API."""
base = '/home/ad.msoe.edu/somed/isaacsim/extscache/omni.anim.people-0.7.9+107.3.3/omni/anim/people'
for fname in ['python_ext.py', 'scripts/character_behavior.py']:
    print(f'\n{"="*60}\n{fname}\n{"="*60}')
    with open(f'{base}/{fname}') as f:
        print(f.read())
