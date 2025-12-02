"""
Configuration settings for HRHUB application.
"""

# App Settings
APP_TITLE = "🏢 HRHUB - HR Matching System"
APP_SUBTITLE = "Bilateral Matching Engine for Candidates & Companies"
VERSION = "1.0.0 - MVP"

# Matching Settings
DEFAULT_TOP_K = 10
MIN_SIMILARITY_SCORE = 0.5
EMBEDDING_DIMENSION = 384

# UI Settings
NETWORK_GRAPH_HEIGHT = 600
TABLE_PAGE_SIZE = 10

# Colors
COLOR_CANDIDATE = "#00FF00"  # Green
COLOR_COMPANY = "#FF0000"    # Red
COLOR_CONNECTION = "#FFFFFF" # White

# Demo Settings (for hardcoded version)
DEMO_CANDIDATE_ID = 0
DEMO_MODE = True  # Set to False when using real data
