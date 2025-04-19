from rouge_score import rouge_scorer
import csv
import os

def compute_rouge_scores(question, context, answer):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = {
        "rougeL_q_ctx": scorer.score(question, context)["rougeL"].fmeasure,
        "rougeL_q_ans": scorer.score(question, answer)["rougeL"].fmeasure,
        "rougeL_ctx_ans": scorer.score(context, answer)["rougeL"].fmeasure,
    }
    return scores

def save_results_to_csv(filepath, results, headers=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers or results[0].keys())
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)