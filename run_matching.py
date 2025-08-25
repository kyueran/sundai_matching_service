# sundai/run_matching.py
import os
import json
import numpy as np
from person_parser import parse_persons_from_csv
from person_vectorizer import PersonVectorizer
from cosine_similarity import cosine_similarity_q1_q2_flipped
from profile_summarizer import make_intro

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_persons_from_files(directory: str, filename: str = "Angle Bracket AB.csv"):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return []

    persons = parse_persons_from_csv(path)

    # Ensure every person has a fallback name
    for i, person in enumerate(persons, start=1):
        if not getattr(person, "name", None):
            person.name = f"Person {i}"

    return persons

def run_matching(persons, output_file="matches.json"):
    vectorizer = PersonVectorizer()
    vectors = [vectorizer.encode_person_fields(p) for p in persons]  

    N = len(persons)
    sim_dict = {}
    results = {}

    for idx in range(N):  # outer loop person index
        sims = []
        for j in range(N):
            if idx == j:
                continue
            if (j, idx) in sim_dict:
                score = sim_dict[(j, idx)]
            else:
                score = cosine_similarity_q1_q2_flipped(vectors[idx], vectors[j])
                sim_dict[(idx, j)] = score
            sims.append((j, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]

        # Collect top matches
        matches = []
        for rank, (j, score) in enumerate(top3, start=1):
            intro = make_intro(persons[idx], persons[j])
            p = persons[j]
            matches.append({
                "username": p.username,
                "name": p.name,
                "linkedin": p.linkedin,
                "role": p.role,
                "intro": intro,
                "ranking": rank,
            })

        # Username of idx as key, matches as content
        results[persons[idx].username] = {
            "self": {
                "username": persons[idx].username,
                "name": persons[idx].name,
                "linkedin": persons[idx].linkedin,
                "role": persons[idx].role,
            },
            "matches": matches
        }

    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {output_file}")
    return results

if __name__ == "__main__":
    persons = load_persons_from_files("profiles")
    if not persons:
        print("No profiles found! Please add person1.txt … person5.txt in profiles/")
    else:
        run_matching(persons)
