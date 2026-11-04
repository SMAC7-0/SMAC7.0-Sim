## Starting up
# Create a virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# install dev dependencies
python -m pip install -e ".[dev]"


## Test Models
# Test Manual Grab
python scripts/test_grab.py

# Test Walk
python scripts/run_gait.py

# View Multiple Cubes
python scripts/multi_block_test.py

# Generate a scene with any number of reusable 3-inch blocks
python scripts/generate_multi_block_scene.py



Inchworm structure
ee1 (root) → [left_wrist, left_hand] → link1 → [top] → link2 → [right_hand, right_wrist] → ee2
