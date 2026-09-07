"""
Natural Language Command & Target Parser for Autonomous Drone Assistant
Extracts target object and intended action (HOVER, LAND, SEARCH) from natural language instructions.
"""

import re

# Object Synonym Mapping to COCO categories
OBJECT_SYNONYMS = {
 'sports ball': ['sports ball', 'ball', 'basketball', 'soccer ball', 'tennis ball'],
 'laptop': ['laptop', 'computer', 'pc', 'notebook'],
 'bottle': ['bottle', 'water bottle', 'drink bottle'],
 'cup': ['cup', 'mug', 'glass'],
 'cell phone': ['cell phone', 'phone', 'mobile phone', 'cellphone', 'smartphone'],
 'chair': ['chair', 'seat', 'armchair'],
 'book': ['book', 'textbook', 'notebook'],
 'mouse': ['mouse'],
 'keyboard': ['keyboard'],
 'scissors': ['scissors'],
 'teddy bear': ['teddy bear', 'bear', 'toy'],
 'apple': ['apple'],
 'banana': ['banana'],
 'backpack': ['backpack', 'bag']
}

def parse_command(user_text):
 """
 Parses a user input string and returns a dict with parsed intent:
 {
 'target': str (COCO category name, e.g. 'sports ball' or 'laptop'),
 'action': str ('HOVER', 'LAND', 'SEARCH', 'TAKEOFF', 'LAND_IMMEDIATE', 'UNKNOWN'),
 'raw_text': str
 }
 """
 text = user_text.strip().lower()
 if not text:
 return {'target': None, 'action': 'NONE', 'raw_text': user_text}

 # Direct immediate commands
 if text in ['takeoff', 'start', 'launch']:
 return {'target': None, 'action': 'TAKEOFF', 'raw_text': user_text}
 if text in ['land', 'stop', 'emergency']:
 return {'target': None, 'action': 'LAND_IMMEDIATE', 'raw_text': user_text}

 # Extract Action Intent
 action = 'HOVER' # Default action
 if re.search(r'\b(land|touchdown|stop)\b', text):
 action = 'LAND'
 elif re.search(r'\b(hover|stay|wait|pause|above)\b', text):
 action = 'HOVER'
 elif re.search(r'\b(search|find|look for|scan)\b', text):
 action = 'SEARCH'

 # Extract Target Object
 target_object = None
 for coco_name, synonyms in OBJECT_SYNONYMS.items():
 for syn in synonyms:
 pattern = r'\b' + re.escape(syn) + r'\b'
 if re.search(pattern, text):
 target_object = coco_name
 break
 if target_object:
 break

 return {
 'target': target_object,
 'action': action if target_object else 'UNKNOWN',
 'raw_text': user_text
 }

# Unit test parser
if __name__ == '__main__':
 test_cases = [
 "go to a ball and hover above it",
 "once you go to a computer, land",
 "find a bottle and hover",
 "go to phone and land",
 "takeoff",
 "land"
 ]
 print("=== Testing Command Parser ===")
 for case in test_cases:
 res = parse_command(case)
 print(f"Input: '{case:35s}' -> Target: {str(res['target']):15s} Action: {res['action']}")
