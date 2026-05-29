"""
HISTORICAL NEWS AI TEST - ChatGPT Revolution

Tests the paradigm shift detector on the AI/ChatGPT revolution that occurred
from late 2022 through 2023. This represents one of the most dramatic paradigm
shifts in recent history.

Expected behavior:
- PRE-CHATGPT: Should show STABLE or low shift score
- EMERGENCE: Should show TRANSITIONAL or early shift signals
- ACCELERATION: Should show strong SHIFT signal
- PARADIGM LOCKED: Should maintain SHIFT signal (new normal established)
"""

import numpy as np
from datetime import datetime
from realtime_paradigm_detector import ParadigmShiftDetector
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("HISTORICAL NEWS TEST: AI/ChatGPT Revolution (2022-2023)")
    print("=" * 80)
    print("\nTesting paradigm shift detector on real headlines from the AI revolution.")
    print("This tests whether the detector can identify one of the most significant")
    print("technological paradigm shifts in recent history.\n")

    # Initialize detector
    detector = ParadigmShiftDetector()

    # Historical time periods with real headlines/sentiment
    time_periods = [
        {
            'label': 'PRE-CHATGPT',
            'date': 'Early 2022',
            'headlines': [
                "AI research continues steady progress",
                "Machine learning improves specific tasks",
                "Chatbots still frustrating to use",
                "Tech companies invest in AI labs",
                "AI art generators gain niche following",
            ],
            'expected': 'STABLE'
        },
        {
            'label': 'EMERGENCE',
            'date': 'Nov-Dec 2022',
            'headlines': [
                "OpenAI releases ChatGPT",
                "New AI chatbot goes viral",
                "Millions sign up for ChatGPT",
                "AI passes professional exams",
                "ChatGPT writes essays, code, poetry",
            ],
            'expected': 'SHIFT/TRANSITIONAL'
        },
        {
            'label': 'ACCELERATION',
            'date': 'Early 2023',
            'headlines': [
                "Microsoft invests billions in OpenAI",
                "Google declares code red over AI",
                "AI disrupting education",
                "Every company racing to add AI",
                "Is this the start of AGI?",
            ],
            'expected': 'SHIFT'
        },
        {
            'label': 'PARADIGM LOCKED',
            'date': 'Mid 2023',
            'headlines': [
                "AI assistants now everywhere",
                "Jobs being automated by AI",
                "AI safety becomes urgent priority",
                "The world has changed forever",
                "No going back to pre-AI world",
            ],
            'expected': 'SHIFT'
        },
    ]

    # Store results for timeline
    results = []

    # Analyze each period
    for period in time_periods:
        print("\n" + "=" * 80)
        print(f"PERIOD: {period['label']} ({period['date']})")
        print("=" * 80)
        print("\nHeadlines:")
        for i, headline in enumerate(period['headlines'], 1):
            print(f"  {i}. {headline}")

        result = detector.analyze(period['headlines'], period['label'])

        print(f"\nExpected: {period['expected']}")
        print(f"Detected: {result['shift_type']}")

        # Store for timeline
        results.append({
            'label': period['label'],
            'date': period['date'],
            'shift_score': result['shift_score'],
            'shift_type': result['shift_type'],
            'expected': period['expected'],
            'top_geodesic': result['top_geodesic'],
            'top_3': result['top_geodesics'][:3],
        })

    # Print timeline summary
    print("\n\n" + "=" * 80)
    print("PARADIGM SHIFT TIMELINE: AI/ChatGPT Revolution")
    print("=" * 80)

    print(f"\n{'Period':<20} {'Date':<15} {'Score':>10} {'Type':<15} {'Top Geodesic':<15}")
    print("-" * 80)

    for r in results:
        print(f"{r['label']:<20} {r['date']:<15} {r['shift_score']:>+10.4f} {r['shift_type']:<15} {r['top_geodesic']:<15}")

    # Analyze score trajectory
    print("\n" + "-" * 80)
    print("SHIFT SCORE EVOLUTION:")
    print("-" * 80)

    max_width = 60
    for r in results:
        score = r['shift_score']
        # Normalize score to bar width (assume range -0.1 to +0.3)
        normalized = (score + 0.1) / 0.4
        bar_width = int(normalized * max_width)
        bar_width = max(0, min(max_width, bar_width))

        bar = '>' * bar_width
        print(f"{r['label']:<20} [{bar:<{max_width}}] {score:+.4f}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    # Check if trajectory matches expected pattern
    score_trajectory = [r['shift_score'] for r in results]

    # Expected pattern: should increase from pre to acceleration
    pre_score = score_trajectory[0]
    emergence_score = score_trajectory[1]
    acceleration_score = score_trajectory[2]
    locked_score = score_trajectory[3]

    print(f"\nScore Trajectory:")
    print(f"  PRE-CHATGPT:      {pre_score:+.4f}")
    print(f"  EMERGENCE:        {emergence_score:+.4f}")
    print(f"  ACCELERATION:     {acceleration_score:+.4f}")
    print(f"  PARADIGM LOCKED:  {locked_score:+.4f}")

    # Calculate changes
    pre_to_emergence = emergence_score - pre_score
    emergence_to_accel = acceleration_score - emergence_score
    accel_to_locked = locked_score - acceleration_score

    print(f"\nScore Changes:")
    print(f"  PRE -> EMERGENCE:        {pre_to_emergence:+.4f}")
    print(f"  EMERGENCE -> ACCELERATION: {emergence_to_accel:+.4f}")
    print(f"  ACCELERATION -> LOCKED:    {accel_to_locked:+.4f}")

    # Success criteria
    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA:")
    print("-" * 80)

    checks = []

    # 1. PRE period should be stable or low
    pre_check = pre_score < 0.1
    checks.append(pre_check)
    status = "PASS" if pre_check else "FAIL"
    print(f"  1. PRE-CHATGPT shows stability (score < 0.1):     {status}")
    print(f"     Actual: {pre_score:+.4f}")

    # 2. Score should increase from PRE to EMERGENCE
    increase_check = emergence_score > pre_score
    checks.append(increase_check)
    status = "PASS" if increase_check else "FAIL"
    print(f"\n  2. Score increases PRE -> EMERGENCE:            {status}")
    print(f"     Change: {pre_to_emergence:+.4f}")

    # 3. ACCELERATION should show strong shift
    accel_check = acceleration_score > 0.05
    checks.append(accel_check)
    status = "PASS" if accel_check else "FAIL"
    print(f"\n  3. ACCELERATION shows strong shift (> 0.05):     {status}")
    print(f"     Actual: {acceleration_score:+.4f}")

    # 4. PARADIGM LOCKED should maintain shift signal
    locked_check = locked_score > 0.0
    checks.append(locked_check)
    status = "PASS" if locked_check else "FAIL"
    print(f"\n  4. PARADIGM LOCKED maintains shift signal:       {status}")
    print(f"     Actual: {locked_score:+.4f}")

    # 5. Overall trajectory should be upward
    trajectory_check = locked_score > pre_score
    checks.append(trajectory_check)
    status = "PASS" if trajectory_check else "FAIL"
    print(f"\n  5. Overall upward trajectory:                    {status}")
    print(f"     Total change: {locked_score - pre_score:+.4f}")

    # Final verdict
    print("\n" + "=" * 80)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        verdict = "EXCELLENT - Detector correctly identified AI paradigm shift"
    elif passed >= total - 1:
        verdict = "GOOD - Detector mostly captured AI paradigm shift"
    elif passed >= total - 2:
        verdict = "FAIR - Detector partially captured AI paradigm shift"
    else:
        verdict = "NEEDS IMPROVEMENT - Detector did not capture AI paradigm shift"

    print(f"OVERALL: {passed}/{total} checks passed")
    print(f"\n{verdict}")
    print("=" * 80)

    # Additional insights
    print("\n" + "=" * 80)
    print("TOP GEODESICS BY PERIOD")
    print("=" * 80)

    for r in results:
        print(f"\n{r['label']} ({r['date']}):")
        for name, sim in r['top_3']:
            print(f"  {name:<15} {sim:.3f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
The AI/ChatGPT revolution represents a clear paradigm shift:

- PRE-CHATGPT: Stable, incremental progress in AI
- EMERGENCE: ChatGPT launches, goes viral, disrupts assumptions
- ACCELERATION: Major tech companies pivot, race begins, fear/excitement
- PARADIGM LOCKED: AI everywhere, world transformed, no going back

This is a textbook paradigm shift that should trigger the detector's
shift geodesics (Earthquake, Death, Wind) while moving away from
stability geodesics (Dog, Deer, Reed).

If the detector correctly identifies this shift, it validates the
paradigm shift signature discovered from 13 historical cases.
""")


if __name__ == "__main__":
    main()
