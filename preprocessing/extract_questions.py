import re
from pathlib import Path
from thefuzz import fuzz, process

def parse_transcript(transcript_text, threshold=80):

    # This object contains the questions which relate to a new numbered item. Follow up sub-questions do not trigger new question-answer block
    incrementor_questions = [
        "How much speech does the subject have now? Do they have phrases with at least three words that they use every day?",
        "Now let me focus on the time when the subject had very little speech. How did they let you know that they wanted something then?",
        "Let me now come back to the subject's talking now. What is their pronunciation like? Are there any words or sounds they don't get quite right? What are they?",
        "Have they ever tended to use rather odd phrases or say the same thing over and over in almost exactly the same way? That is, either phrases they have heard other people use or ones they have made up.",
        "Did they ever use odd phrases more often than they do now?",
        "Can you have a conversation with the subject? That is, if you say something to them, without asking a direct question, what will the subject usually do?",
        "Are there times when the subject uses socially inappropriate questions or statements? For example, do they regularly ask personal questions or make personal comments at awkward times?",
        "Has the subject ever got their personal pronouns the wrong way round? For example, has the subject ever mixed up 'you' and 'I'?",
    ]
    
    lines = transcript_text.strip().splitlines()
    qa_pairs = {}
    current_question = None
    current_answer = []
    question_index = 0

    for line in lines:
        match = re.match(r"^(Speaker \d+):\s*(.+)", line.strip())
        if not match:
            continue

        speaker, text = match.groups()
        text = text.strip()

        if speaker == "Speaker 1":
            if question_index < len(incrementor_questions):
                match_text, score = process.extractOne(text, incrementor_questions[question_index:], scorer=fuzz.token_set_ratio)

            if score >= threshold:
                if current_question:
                    qa_pairs[question_index + 29] = (current_question, " ".join(current_answer).strip())
                    current_answer = []

                current_question = match_text
                question_index += 1
            else:
                current_answer.append(f"Speaker 1: {text}")

        elif speaker == "Speaker 2":
                    current_answer.append(text)

    # Final question
    if current_question:
        qa_pairs[question_index + 29] = (current_question, " ".join(current_answer).strip())

    return qa_pairs

# Use pathlib for cross-platform file handling
file_path = Path("preprocessing") / "example_transcript.txt"
with file_path.open("r", encoding="utf-8") as f:
    text = f.read()

result = parse_transcript(text)

for q_num in sorted(result.keys()):
    question, answer = result[q_num]
    print(f"\nQuestion {q_num}")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 60)