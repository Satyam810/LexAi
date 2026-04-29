from src.query_validator import validate_query

tests = [
    ("Empty",          "",                                                          False),
    ("Single word",    "bail",                                                      False),
    ("Below min",      "IPC 302",                                                   False),
    ("Non-legal",      "what is the weather like in Mumbai today please tell me",   False),
    ("Hindi",          "आरोपी ने हत्या की थी और न्यायालय ने दोषी ठहराया",         False),
    ("Too long",       "accused " * 1000,                                           False),
    ("Valid murder",   "Accused charged under IPC Section 302 for murder. Prosecution relies on eyewitness testimony and forensic DNA evidence. Defence claims alibi.", True),
    ("Valid bail",     "Applicant seeks bail in Sessions Court. Arrested under IPC 420 for cheating. No prior conviction. FIR filed by complainant.", True),
    ("Valid appeal",   "Appeal against conviction under IPC 376. Sessions judge dismissed bail. High Court hearing.", True),
]

passed = failed = 0
for label, query, expect_valid in tests:
    is_valid, error = validate_query(query)
    ok = (is_valid == expect_valid)
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else:  failed += 1
    detail = f"valid={is_valid}"
    if not is_valid:
        detail += f" | {error[:70]}"
    print(f"  {status}  {label:<20}  {detail}")

print(f"\nValidator: {passed} PASS / {failed} FAIL")
assert failed == 0, "Validator still has failures — do not proceed to Fix 5"
