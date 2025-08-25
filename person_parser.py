# sundai/person_parser.py
import csv
import re
from person_vectorizer import Person

def normalize_header(h: str) -> str:
    # Replace all whitespace (\r, \n, tabs, spaces) with a single space
    h = re.sub(r"\s+", " ", h.strip())
    # Normalize curly quotes
    h = h.replace("’", "'")
    return h

def parse_persons_from_csv(csv_file: str):
    persons = []
    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Normalize headers
        reader.fieldnames = [normalize_header(h) for h in reader.fieldnames]

        for row in reader:
            clean_row = {normalize_header(k): (v or "").strip() for k, v in row.items()}

            person = Person(
                username=clean_row.get("Username", ""),
                name=clean_row.get("Name", ""),
                linkedin=clean_row.get("LinkedIn", ""),  # trailing space cleaned
                role=clean_row.get("What are you looking for?", ""),
                q1=clean_row.get("What are you an expert in? Separate multiple topics by commas", ""),
                q2=clean_row.get("What do you want to learn? Separate multiple topics by commas", ""),
                q3=clean_row.get("What's something you've enjoyed doing outside of work in the past month?", ""),
                q4=clean_row.get("What do you want to achieve in the coming month?", "")
            )
            persons.append(person)
    return persons