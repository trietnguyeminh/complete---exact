import re, sys
sys.stdout.reconfigure(encoding='utf-8')

# ===== Copy of the matching functions from the notebook =====

PHYSICS_TOLERANCE = 0.05

def _normalize_number(s):
    s = str(s).strip()
    s = re.sub(r'\s*[A-Za-z/]+\s*$', '', s).strip()
    m = re.match(
        r'^([+-]?[\d.]+)\s*[×xX\*]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?\s*$',
        s
    )
    if m:
        try:
            mantissa = float(m.group(1))
            exp = int(m.group(2))
            return mantissa * (10 ** exp)
        except:
            pass
    try:
        return float(s)
    except:
        pass
    return None

def _physics_answer_match(predicted, ground_truth, tolerance=PHYSICS_TOLERANCE):
    p = str(predicted).strip()
    g = str(ground_truth).strip()
    if p.lower() == g.lower():
        return True
    p_num = _normalize_number(p)
    g_num = _normalize_number(g)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return abs(p_num) < 1e-9
        rel_error = abs(p_num - g_num) / abs(g_num)
        return rel_error <= tolerance
    p_low = p.lower()
    g_low = g.lower()
    if len(g_low) > 3 and (g_low in p_low or p_low in g_low):
        return True
    p_clean = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', p)
    g_clean = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', g)
    if p_clean and g_clean:
        p2 = _normalize_number(p_clean)
        g2 = _normalize_number(g_clean)
        if p2 is not None and g2 is not None and g2 != 0:
            if abs(p2 - g2) / abs(g2) <= tolerance:
                return True
    return False

# ===== TEST CASES =====
tests = [
    # (predicted, ground_truth, expected_match)
    # Exact match
    ("45", "45", True),
    ("0.05", "0.05", True),
    ("100", "100", True),
    
    # Numeric tolerance (within 5%)
    ("44.5", "45", True),       # 1.1% error
    ("47", "45", True),         # 4.4% error
    ("43", "45", True),         # 4.4% error  
    ("42", "45", False),        # 6.7% error -> too much
    
    # Scientific notation
    ("0.02445", "24.45 × 10^-3", True),
    ("24.45e-3", "24.45 × 10^-3", True),
    ("0.024", "24.45 × 10^-3", True),   # within 5%
    
    # Model output with units
    ("45 J", "45", True),
    ("0.05 N", "0.05", True),
    ("100 μF", "100", True),
    
    # Text answers
    ("maximum", "maximum", True),
    ("The voltage is halved", "the voltage is halved", True),
    ("decreases by half", "decreases by half", True),
    
    # Text containment
    ("The answer is maximum value", "maximum", True),
    
    # Mismatches
    ("A", "45", False),
    ("Unknown", "0.05", False),
    ("Yes", "100", False),
    ("B", "C", False),
]

print(f"Running {len(tests)} test cases...")
print(f"{'Predicted':>30} | {'Ground Truth':>25} | {'Expected':>8} | {'Got':>8} | Status")
print("-" * 95)

passed = 0
failed = 0
for pred, gt, expected in tests:
    result = _physics_answer_match(pred, gt)
    ok = result == expected
    status = "✓" if ok else "✗ FAIL"
    print(f"{pred:>30} | {gt:>25} | {str(expected):>8} | {str(result):>8} | {status}")
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{passed}/{len(tests)} passed, {failed} failed")

# Also test _normalize_number
print("\n=== _normalize_number tests ===")
norm_tests = [
    ("45", 45.0),
    ("0.05", 0.05),
    ("24.45 × 10^-3", 0.02445),
    ("5.234 × 10^-3", 0.005234),
    ("6.4 × 10^5", 640000.0),
    ("45 J", 45.0),
    ("100 μF", 100.0),
]
for s, expected in norm_tests:
    got = _normalize_number(s)
    ok = got is not None and abs(got - expected) < 1e-10
    status = "✓" if ok else f"✗ (got {got})"
    print(f"  {s:>25} -> {got} (expected {expected}) {status}")
