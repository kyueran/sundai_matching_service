import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def make_intro(my_info, other_info):
    concat_my_info = " ".join([
        my_info.q1, my_info.q2, my_info.q3, my_info.q4
    ])

    concat_other_info = " ".join([
        other_info.q1, other_info.q2, other_info.q3, other_info.q4
    ])

    prompt = f"""
Here are two people’s self-descriptions.

{other_info.name}'s description:
{concat_other_info}

Write a short, friendly 1–2 sentence introduction for them, starting with their name.

Now here is my description:
{concat_my_info}

If there are overlapping interests, keywords, or activities between me and this person,
add ONE short sentence that starts with "Fun fact: Both of you ..." and naturally uses the shared word or phrase.
If there is no overlap, only return the introduction.
"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()
