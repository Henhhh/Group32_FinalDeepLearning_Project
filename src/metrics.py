import re

def normalize_em(text):
    return re.sub('\\s+', ' ', str(text).lower().strip())

def normalize_anls(text):
    return str(text).lower().strip()

def levenshtein(a, b):
    """Minimum number of character insertions, deletions, and substitutions."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]

def calculate_anls(prediction, answers):
    prediction = normalize_anls(prediction)
    scores = []
    for answer in answers:
        answer = normalize_anls(answer)
        max_length = max(len(prediction), len(answer))
        if max_length == 0:
            score = 1.0
        else:
            normalized_distance = levenshtein(prediction, answer) / max_length
            score = 1.0 - normalized_distance if normalized_distance < 0.5 else 0.0
        scores.append(score)
    return max(scores, default=0.0)

def score_row(row):
    return {**row, 'correct': any(normalize_em(row['prediction']) == normalize_em(a)
                                 for a in row['answers']),
            'anls': calculate_anls(row['prediction'], row['answers'])}

def summarize(rows):
    if not rows:
        raise ValueError('No predictions to summarize')
    scored = [score_row(r) for r in rows]
    n = len(scored)
    return {'questions': n, 'correct': sum(r['correct'] for r in scored),
            'em_percent': 100 * sum(r['correct'] for r in scored) / n,
            'anls': sum(r['anls'] for r in scored) / n,
            'seconds_per_question': sum(r['seconds'] for r in scored) / n}
