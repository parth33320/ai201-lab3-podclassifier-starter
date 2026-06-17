import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation() -> dict:
    """
    Run the classifier against the held-out test set and return full results.

    This function is already complete. It:
      1. Loads the labeled training examples (from your my_labels.json)
      2. Loads the test episodes (with ground-truth labels)
      3. Runs classify_episode() on each test description
      4. Returns a results dict with predictions, ground truth, and per-episode detail

    You'll use the results dict in compute_accuracy() and compute_per_class_accuracy().
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(episode["title"], episode["description"], labeled_examples)
        results.append({
            "id": episode["id"],
            "title": episode["title"],
            "description": episode["description"],
            "ground_truth": episode["label"],
            "predicted": prediction["label"],
            "reasoning": prediction["reasoning"],
            "correct": prediction["label"] == episode["label"],
        })

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.
    Handles empty lists (0.0) and mismatched lengths (ValueError).
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("mismatched length")

    if not predictions:
        return 0.0

    correct = 0
    for p, g in zip(predictions, ground_truth):
        if p == g:
            correct += 1

    return correct / len(predictions)


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class in VALID_LABELS.
    Initializes all VALID_LABELS to 0.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("mismatched length")

    # Initialize all valid labels with zero counts
    stats = {}
    for label in VALID_LABELS:
        stats[label] = {"correct": 0, "total": 0, "accuracy": 0.0}

    for p, g in zip(predictions, ground_truth):
        if g in stats:
            stats[g]["total"] += 1
            if p == g:
                stats[g]["correct"] += 1

    for label in stats:
        if stats[label]["total"] > 0:
            stats[label]["accuracy"] = stats[label]["correct"] / stats[label]["total"]

    return stats


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.

    This function is already complete. Pass it the dict returned by run_evaluation().
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (10 - int(stats["accuracy"] * 10))
        lines.append(f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']})")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(f"  [{r['ground_truth']} → {r['predicted']}] {r['title']}")
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)


def test_evaluation_math():
    """
    Test the evaluation functions with various cases.
    """
    # 1. Standard 2/4 case
    predictions = ["interview", "solo", "unknown", "panel"]
    ground_truth = ["interview", "solo", "narrative", "narrative"]
    accuracy = compute_accuracy(predictions, ground_truth)
    assert accuracy == 0.5, f"Expected 0.5 accuracy, got {accuracy}"

    per_class = compute_per_class_accuracy(predictions, ground_truth)
    assert per_class["interview"]["correct"] == 1
    assert per_class["interview"]["total"] == 1
    assert per_class["narrative"]["correct"] == 0
    assert per_class["narrative"]["total"] == 2
    assert per_class["panel"]["total"] == 0  # Should be 0 but present

    # 2. Empty list case
    assert compute_accuracy([], []) == 0.0

    # 3. Mismatched length case
    try:
        compute_accuracy(["solo"], ["solo", "interview"])
        assert False, "Should have raised ValueError for mismatched lengths"
    except ValueError:
        pass

    print("✓ test_evaluation_math passed!")


if __name__ == "__main__":
    test_evaluation_math()
