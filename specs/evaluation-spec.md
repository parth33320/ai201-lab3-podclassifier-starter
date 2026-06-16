# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (Number of correct predictions) / (Total number of predictions)

A prediction is "correct" if it exactly matches the ground-truth label.
Note: "unknown" matches are always counted as incorrect, even if both are "unknown".
```

---

**Step-by-step logic:**

```
1. Validate that predictions and ground_truth have the same length. Raise ValueError if not.
2. If the lists are empty, return 0.0.
3. Validate that all labels in ground_truth are in VALID_LABELS. Raise ValueError if not.
4. Count the number of indices where predictions[i] == ground_truth[i] AND predictions[i] is in VALID_LABELS.
5. Divide this count by the total number of items and return as a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. This avoids a division-by-zero error and represents a "no data" state gracefully.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Matches:
- Index 0: interview == interview (Correct)
- Index 1: solo == solo (Correct)
- Index 2: panel != solo (Incorrect)
- Index 3: interview != narrative (Incorrect)

Total correct: 2
Total predictions: 4
Accuracy: 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode is "correct" for a class (e.g., "interview") if its ground-truth label is "interview"
AND the predicted label is also "interview".
```

---

**What does "total" mean for a given class?**

```
"total" is the number of episodes in the ground_truth list that have that specific label.
```

---

**Step-by-step logic:**

```
1. Validate that predictions and ground_truth have the same length. Raise ValueError if not.
2. Validate that all labels in ground_truth are in VALID_LABELS. Raise ValueError if not.
3. Initialize a dictionary with entries for each label in VALID_LABELS, starting at:
   {"correct": 0, "total": 0, "accuracy": 0.0}
4. Loop through zip(predictions, ground_truth):
   - For each (pred, truth):
     - Increment total for the class 'truth'.
     - If pred == truth, increment correct for the class 'truth'.
5. Loop through each label in VALID_LABELS in the results dictionary:
   - If total > 0, set accuracy = correct / total.
   - Else, set accuracy = 0.0.
6. Return the dictionary.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
The accuracy should be set to 0.0. This explicitly tells the user that this class was
not represented in the current test set.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
