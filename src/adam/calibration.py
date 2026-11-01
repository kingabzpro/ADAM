"""Diagnostics for independently labeled D3 classification decisions."""
import math

from adam.decisions import Decision

LABELS = ('BUY', 'HOLD', 'SELL')


def evaluate_calibration(examples, *, bins=10):
    if type(bins) is not int or not 1 <= bins <= 100:
        raise ValueError('bins must be an integer between 1 and 100')
    groups = [{'lower': i / bins, 'upper': (i + 1) / bins, 'count': 0,
               'confidence_sum': 0.0, 'correct_sum': 0} for i in range(bins)]
    included, excluded, correct = 0, 0, 0
    brier, log_loss = 0.0, 0.0
    agent = None
    for example in examples:
        if not isinstance(example, dict) or set(example) != {'decision', 'label'} or example['label'] not in LABELS:
            raise ValueError('Each example must contain a decision and a valid independent label')
        decision = Decision.model_validate(example['decision'])
        if decision.status != 'completed' or not decision.reliable or decision.stale:
            excluded += 1
            continue
        if agent is not None and agent != decision.agent:
            raise ValueError('Evaluate one agent category at a time')
        agent = decision.agent
        scores = decision.probabilities
        confidence = scores[decision.signal]
        if confidence < max(scores.values()):
            raise ValueError('Decision signal must have a highest score')
        outcome = example['label']
        is_correct = decision.signal == outcome
        included += 1
        correct += is_correct
        brier += sum((scores[label] - (label == outcome)) ** 2 for label in LABELS)
        log_loss -= math.log(max(scores[outcome], 1e-15))
        group = groups[min(int(confidence * bins), bins - 1)]
        group['count'] += 1
        group['confidence_sum'] += confidence
        group['correct_sum'] += is_correct
    if not included:
        raise ValueError('At least one reliable, non-stale completed decision is required')
    ece = 0.0
    output = []
    for group in groups:
        count = group['count']
        mean = group['confidence_sum'] / count if count else None
        accuracy = group['correct_sum'] / count if count else None
        if count:
            ece += count / included * abs(mean - accuracy)
        output.append({'lower': group['lower'], 'upper': group['upper'], 'count': count,
                       'mean_confidence': mean, 'accuracy': accuracy})
    return {'agent': agent, 'samples': included, 'excluded': excluded, 'accuracy': correct / included,
            'brier_score': brier / included, 'log_loss': log_loss / included,
            'expected_calibration_error': ece, 'bins': output}


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='Evaluate independently labeled D3 decision scores from JSON Lines.')
    parser.add_argument('path', help='UTF-8 JSONL file containing decision and label fields')
    parser.add_argument('--bins', type=int, default=10)
    args = parser.parse_args()
    try:
        with open(args.path, encoding='utf-8') as file:
            examples = [json.loads(line) for line in file if line.strip()]
        result = evaluate_calibration(examples, bins=args.bins)
    except (OSError, ValueError, TypeError) as error:
        print(f'Calibration failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, allow_nan=False))
