import sys
from pathlib import Path

target = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/code/io_utils.py")
content = target.read_text(encoding="utf-8")

# Fix aliases
old_aliases = """_REVIEW_COLUMN_ALIASES = [
    "ReviewText", "review_text", "text", "Text", "text_",
    "comment", "Comment", "body", "Body",
    "content", "Content", "review_body", "ReviewBody",
]"""
new_aliases = """_REVIEW_COLUMN_ALIASES = [
    "review", "Review", "ReviewText", "review_text", "text", "Text", "text_",
    "comment", "Comment", "body", "Body",
    "content", "Content", "review_body", "ReviewBody",
]"""
if old_aliases in content:
    content = content.replace(old_aliases, new_aliases)
else:
    # Try with different whitespace if needed
    content = content.replace('"ReviewText", "review_text"', '"review", "Review", "ReviewText", "review_text"')

# Fix mapping loop
old_loop = """    for alias in _REVIEW_COLUMN_ALIASES:
        if alias in frame.columns and "text" not in frame.columns:
            frame["text"] = frame[alias]
            break"""
new_loop = """    for alias in _REVIEW_COLUMN_ALIASES:
        if alias in frame.columns and "text" not in frame.columns:
            frame["text"] = frame[alias]
            if alias != "text":
                frame.drop(columns=[alias], inplace=True)
            break"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
else:
    # Manual surgical fix
    content = content.replace('frame["text"] = frame[alias]', 'frame["text"] = frame[alias]\n            if alias != "text":\n                frame.drop(columns=[alias], inplace=True)')

target.write_text(content, encoding="utf-8")
print("Done fixing io_utils.py")
