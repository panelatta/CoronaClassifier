import re

def extract_pattern(s):
    # Regex pattern: one or more digits followed by one or more uppercase letters
    pattern = r'\d+[A-Z]+'

    # Search for the pattern in the string
    match = re.search(pattern, s)
    if match:
        return match.group()  # Returns the matched part of the string
    else:
        return None  # No match found

# Test cases
strings = ["21E (Theta)", "21A (Delta)", "20B"]
extracted = [extract_pattern(s) for s in strings]

print(extracted)
