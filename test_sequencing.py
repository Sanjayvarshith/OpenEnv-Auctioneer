"""
test_sequencing.py — Unit tests for the HardSequencingGrader and DP oracle.

Run: python test_sequencing.py
"""

import sys
sys.path.insert(0, ".")

from environment import HardSequencingGrader


def test_dp_oracle_simple():
    """Fixed scenario: 4 steps, known optimal is to bid on steps 0 and 2."""
    g = HardSequencingGrader()

    # Step 0: cheap (clears at $1), good CTR 0.10
    g.record_step(step=0, context="Fitness", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    # Step 1: expensive (clears at $8), low CTR 0.02
    g.record_step(step=1, context="Tech", clearing_price=8.0,
                  base_ctr=0.02, auction_won=False, cost=0.0,
                  conversion_value=15.0)
    # Step 2: cheap (clears at $1), good CTR 0.10
    g.record_step(step=2, context="Fashion", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    # Step 3: cheap (clears at $1), good CTR 0.08
    g.record_step(step=3, context="Gaming", clearing_price=1.0,
                  base_ctr=0.08, auction_won=True, cost=1.0,
                  conversion_value=15.0)

    oracle_val = g._oracle_conversions(budget=10.0)
    assert oracle_val > 0, f"Oracle should find positive conversions, got {oracle_val}"
    print(f"  DP oracle conversions: {oracle_val:.4f} ✓")

    score = g.episode_score(initial_budget=10.0)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  Episode score: {score:.4f} ✓")


def test_never_bid_score_zero():
    """Agent never bids → score should be 0.0."""
    g = HardSequencingGrader()
    for i in range(24):
        ctx = ["Fitness", "Tech", "Fashion", "Gaming"][i % 4]
        g.record_step(step=i, context=ctx, clearing_price=1.0,
                      base_ctr=0.05, auction_won=False, cost=0.0,
                      conversion_value=15.0)

    score = g.episode_score(initial_budget=100.0)
    assert score == 0.0, f"Expected 0.0 for never-bid agent, got {score}"
    print(f"  Never-bid score: {score} ✓")


def test_diversity_multiplier():
    """Check 1.2× when ≥3 contexts, 1.0 otherwise."""
    g = HardSequencingGrader()

    # Only 2 contexts
    g.record_step(step=0, context="Fitness", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    g.record_step(step=1, context="Tech", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    assert g._diversity_multiplier() == 1.0, "Should be 1.0 with only 2 contexts"

    # Add 3rd context
    g.record_step(step=2, context="Fashion", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    assert g._diversity_multiplier() == 1.20, "Should be 1.2 with 3 contexts"
    print(f"  Diversity multiplier: correct ✓")


def test_carryover_schedule():
    """Verify carry-over boosts decay correctly: +15%/+10%/+5%."""
    g = HardSequencingGrader()

    # Win at step 0, skip rest
    g.record_step(step=0, context="Fitness", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    for i in range(1, 5):
        g.record_step(step=i, context="Tech", clearing_price=1.0,
                      base_ctr=0.10, auction_won=False, cost=0.0,
                      conversion_value=15.0)

    agent_conv = g._agent_conversions()
    # Only step 0 won, no prior wins → no boost, base conversion = 0.10 * 15 = 1.5
    expected = 0.10 * 15.0
    assert abs(agent_conv - expected) < 0.001, \
        f"Expected {expected}, got {agent_conv}"
    print(f"  Carry-over schedule: correct ✓")

    # Now test with consecutive wins
    g.reset()
    g.record_step(step=0, context="Fitness", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)
    g.record_step(step=1, context="Tech", clearing_price=1.0,
                  base_ctr=0.10, auction_won=True, cost=1.0,
                  conversion_value=15.0)

    agent_conv = g._agent_conversions()
    # step 0: no boost → 0.10 * 15 = 1.5
    # step 1: boost from step 0 win = +15% → 0.10 * 1.15 * 15 = 1.725
    expected = 1.5 + 0.10 * 1.15 * 15.0
    assert abs(agent_conv - expected) < 0.001, \
        f"Expected {expected}, got {agent_conv}"
    print(f"  Consecutive carry-over: correct ✓")


def test_episode_score_bounded():
    """Score should always be in [0.0, 1.0]."""
    g = HardSequencingGrader()

    # Agent bids on everything (cheap auctions)
    for i in range(24):
        ctx = ["Fitness", "Tech", "Fashion", "Gaming"][i % 4]
        g.record_step(step=i, context=ctx, clearing_price=0.50,
                      base_ctr=0.08, auction_won=True, cost=0.50,
                      conversion_value=15.0)

    score = g.episode_score(initial_budget=100.0)
    assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
    print(f"  All-bid score: {score:.4f} (bounded) ✓")


def test_empty_episode():
    """Empty episode → score 0.0."""
    g = HardSequencingGrader()
    score = g.episode_score(initial_budget=100.0)
    assert score == 0.0, f"Expected 0.0 for empty episode, got {score}"
    print(f"  Empty episode: {score} ✓")


if __name__ == "__main__":
    tests = [
        ("DP Oracle (simple scenario)", test_dp_oracle_simple),
        ("Never-bid agent → 0.0", test_never_bid_score_zero),
        ("Diversity multiplier", test_diversity_multiplier),
        ("Carry-over schedule", test_carryover_schedule),
        ("Score bounded [0, 1]", test_episode_score_bounded),
        ("Empty episode", test_empty_episode),
    ]

    print("=" * 52)
    print("  HardSequencingGrader Unit Tests")
    print("=" * 52)

    passed = 0
    for name, fn in tests:
        try:
            print(f"\n▸ {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'─'*52}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'─'*52}")

    if passed < len(tests):
        sys.exit(1)
