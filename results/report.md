# BeamDial results (410 tasks x 5 reasoning levels)

Prices used: $0.4/M input, $1.2/M output tokens.

## Policies

| Policy | Accuracy | Tokens / task | $ / 1k tasks | Calls / task | vs fixed max |
|---|---|---|---|---|---|
| fixed: low | 96.3% | 914 | $1.16 | 1.00 | -0.2 pts, 55% of tokens |
| fixed: medium | 96.1% | 1,236 | $1.54 | 1.00 | -0.5 pts, 75% of tokens |
| fixed: high | 97.3% | 1,471 | $1.82 | 1.00 | +0.7 pts, 89% of tokens |
| fixed: xhigh | 97.3% | 1,603 | $1.98 | 1.00 | +0.7 pts, 97% of tokens |
| fixed: max | 96.6% | 1,657 | $2.05 | 1.00 | +0.0 pts, 100% of tokens |
| oracle (upper bound) | 98.0% | 816 | $1.04 | 1.00 | +1.5 pts, 49% of tokens |
| cascade from low (check-based) | 96.8% | 965 | $1.22 | 1.03 | +0.2 pts, 58% of tokens |
| cascade from medium (check-based) | 96.1% | 1,280 | $1.60 | 1.02 | -0.5 pts, 77% of tokens |
| agreement cascade | 97.6% | 2,276 | $2.84 | 1.82 | +1.0 pts, 137% of tokens |
| low, high if low thinks >2000 tok | 97.6% | 1,731 | $2.14 | 1.12 | +1.0 pts, 104% of tokens |
| low, high if low thinks >3000 tok | 97.3% | 1,455 | $1.81 | 1.06 | +0.7 pts, 88% of tokens |
| low, high if low thinks >4000 tok | 97.1% | 1,259 | $1.57 | 1.03 | +0.5 pts, 76% of tokens |
| router cv5 @ 0.8 | 96.3% | 914 | $1.16 | 1.00 | -0.2 pts, 55% of tokens |
| router cv5 @ 0.9 | 96.6% | 964 | $1.22 | 1.00 | +0.0 pts, 58% of tokens |
| router cv5 + check @ 0.8 | 96.8% | 965 | $1.22 | 1.03 | +0.2 pts, 58% of tokens |
| router cv5 + check @ 0.9 | 96.6% | 1,012 | $1.28 | 1.03 | +0.0 pts, 61% of tokens |
| router loso @ 0.8 | 96.3% | 914 | $1.16 | 1.00 | -0.2 pts, 55% of tokens |
| router loso @ 0.9 | 96.1% | 920 | $1.16 | 1.00 | -0.5 pts, 56% of tokens |
| router loso + check @ 0.8 | 96.8% | 965 | $1.22 | 1.03 | +0.2 pts, 58% of tokens |
| router loso + check @ 0.9 | 96.6% | 967 | $1.22 | 1.03 | +0.0 pts, 58% of tokens |

## Accuracy by level and task group

| Group | n | low | medium | high | xhigh | max |
|---|---|---|---|---|---|---|
| GSM8K (easy math) | 40 | 97.5% | 97.5% | 97.5% | 97.5% | 97.5% |
| MATH L1-5 | 160 | 100.0% | 98.8% | 99.4% | 100.0% | 99.4% |
| AMC / AIME 2024 / Olympiad | 110 | 94.5% | 95.5% | 98.2% | 96.4% | 95.5% |
| HumanEval | 50 | 100.0% | 98.0% | 96.0% | 100.0% | 98.0% |
| MBPP | 50 | 84.0% | 86.0% | 90.0% | 88.0% | 88.0% |

Excluding 6 tasks with broken reference answers or tests (see data/flagged_tasks.json), 404 tasks: low 97.8%, medium 97.5%, high 98.8%, xhigh 98.8%, max 98.0%

## Median completion tokens by level

| Group | low | medium | high | xhigh | max |
|---|---|---|---|---|---|
| GSM8K (easy math) | 238 | 220 | 202 | 174 | 176 |
| MATH L1-5 | 365 | 430 | 385 | 348 | 313 |
| AMC / AIME 2024 / Olympiad | 1,136 | 1,597 | 1,517 | 1,756 | 1,500 |
| HumanEval | 418 | 520 | 386 | 327 | 425 |
| MBPP | 374 | 457 | 362 | 440 | 472 |

## Cheapest level that solved each task

low: 395, medium: 6, none: 8, high: 1
