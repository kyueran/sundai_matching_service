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

    for i in range(N):
        sims = []
        for j in range(N):
            if i == j:
                continue
            if (j, i) in sim_dict:
                score = sim_dict[(j, i)]
            else:
                score = cosine_similarity_q1_q2_flipped(vectors[i], vectors[j])
                sim_dict[(i, j)] = score
            sims.append((j, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]
        # Collect top matches
        matches = []
        for i, (j, score) in enumerate(top3):
            intro = make_intro(persons[i], persons[j])
            p = persons[j]
            matches.append({
                "username": p.username,
                "name": p.name,
                "linkedin": p.linkedin,
                "role": p.role,
                "intro": intro,
                "ranking": i + 1,
            })
        
        # Username of i as key, matches as content
        results[persons[i].username] = {
            "self": {
                "username": persons[i].username,
                "name": persons[i].name,
                "linkedin": persons[i].linkedin,
                "role": persons[i].role,
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
