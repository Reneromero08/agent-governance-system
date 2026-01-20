# Catalytic Context Stress Test Brief

## Objective

Stress-test the catalytic context system with REAL use cases that challenge:
1. **Semantic drift** - Query at turn 90 has no keyword overlap with turn 3
2. **Implicit references** - Must understand meaning, not just match keywords
3. **Dense information** - Multiple facts per turn (500+ tokens)
4. **Interleaved topics** - Constant context switching (A-B-C-A-D-B pattern)
5. **Accumulating constraints** - Later decisions depend on earlier ones

**Target**: 1.2B parameter model (liquid/lfm2.5-1.2b) with 32K context window

**Success Criteria**: System surfaces original fact in context at recall query (not LLM hallucination)

---

## Test Configuration

```python
LLM_STUDIO_BASE = "http://10.5.0.2:1234"
MODEL = "liquid/lfm2.5-1.2b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CONTEXT_WINDOW = 32768
E_THRESHOLD = 0.3  # Born rule threshold for retrieval
```

---

## Use Case 1: Software Architecture Session (RECOMMENDED)

### Scenario
Building a fintech payment API. 100-turn session covering requirements through deployment.

### Phase Structure

| Phase | Turns | Topic | Key Facts to Plant |
|-------|-------|-------|-------------------|
| 1. Requirements | 1-10 | Business constraints | Auth: JWT with RS256, Rate limit: 100/min, DB: Postgres |
| 2. Data Model | 11-20 | Schema design | Transaction table has 12 columns, Idempotency key is UUID v4 |
| 3. API Design | 21-35 | Endpoints | POST /payments uses ISO 8601 timestamps, Max payload 1MB |
| 4. Security | 36-45 | Auth/authz | API keys are 32-char hex, Webhook signatures use HMAC-SHA256 |
| 5. Error Handling | 46-55 | Edge cases | Timeout is 30 seconds, Retry policy: 3x with exponential backoff |
| 6. Testing | 56-65 | Test strategy | Coverage target 85%, Load test: 10K TPS |
| 7. Integration | 66-75 | Third-party | Stripe webhook endpoint /hooks/stripe, PCI compliance level 1 |
| 8. Deployment | 76-85 | Infrastructure | Kubernetes cluster: 3 nodes, Pod memory limit: 512MB |
| 9. Monitoring | 86-92 | Observability | Alert threshold: p99 > 500ms, Log retention: 30 days |
| 10. Recall Tests | 93-100 | Verification | Query planted facts from turns 1-45 |

### Planted Facts (15 total)

```
Turn 3:  "Authentication must use JWT tokens signed with RS256 algorithm.
         Keys rotate every 90 days. Maximum session duration is 24 hours."
         Keywords: [RS256, 90 days, 24 hours]

Turn 7:  "Rate limiting is set to 100 requests per minute per API key.
         Burst allowance is 20 requests. Throttled responses return 429."
         Keywords: [100 requests, minute, 429]

Turn 12: "The transactions table schema has 12 columns: id, amount, currency,
         status, created_at, updated_at, merchant_id, customer_id,
         idempotency_key, metadata, fee_amount, settlement_date."
         Keywords: [12 columns, idempotency_key, settlement_date]

Turn 18: "Idempotency keys must be UUID v4 format. They expire after 24 hours.
         Duplicate requests within window return cached response."
         Keywords: [UUID v4, 24 hours, cached]

Turn 25: "POST /v1/payments endpoint accepts JSON with required fields:
         amount (integer cents), currency (ISO 4217), source (payment method).
         Timestamps use ISO 8601 format with timezone."
         Keywords: [integer cents, ISO 4217, ISO 8601]

Turn 31: "Maximum request payload is 1MB. Requests exceeding this return 413.
         Response bodies are limited to 5MB for batch operations."
         Keywords: [1MB, 413, 5MB]

Turn 38: "API keys are 32-character hexadecimal strings prefixed with 'pk_'
         for publishable and 'sk_' for secret keys."
         Keywords: [32-character, pk_, sk_]

Turn 42: "Webhook signatures use HMAC-SHA256 with a shared secret.
         Signature header is X-Signature-256. Tolerance window is 5 minutes."
         Keywords: [HMAC-SHA256, X-Signature-256, 5 minutes]

Turn 48: "Request timeout is 30 seconds for synchronous operations.
         Async operations have 5-minute timeout with polling endpoint."
         Keywords: [30 seconds, 5-minute, polling]

Turn 52: "Retry policy: maximum 3 attempts with exponential backoff.
         Base delay 1 second, max delay 30 seconds. Jitter of +/-10%."
         Keywords: [3 attempts, exponential, 30 seconds]

Turn 58: "Code coverage target is 85% for unit tests. Integration tests
         must cover all payment flows. Mutation testing score > 70%."
         Keywords: [85%, mutation, 70%]

Turn 63: "Load testing target: sustain 10,000 TPS for 10 minutes.
         p99 latency must stay under 200ms during load test."
         Keywords: [10,000 TPS, 10 minutes, 200ms]

Turn 68: "Stripe webhook endpoint is /v1/hooks/stripe. Events are queued
         in Redis before processing. Dead letter queue after 5 failures."
         Keywords: [/v1/hooks/stripe, Redis, 5 failures]

Turn 73: "PCI DSS compliance level 1. Card data never touches our servers.
         Tokenization via Stripe. Annual audit by QSA required."
         Keywords: [level 1, tokenization, QSA]

Turn 78: "Kubernetes deployment: 3 nodes, each 4 vCPU / 16GB RAM.
         Pod resource limits: 512MB memory, 0.5 CPU. HPA scales 3-10 pods."
         Keywords: [3 nodes, 512MB, 3-10 pods]
```

### Recall Queries (No keyword overlap)

```
Turn 93: "For the security audit, what signing algorithm did we choose for tokens?"
         Expected: RS256 (from turn 3)

Turn 94: "What's our throttling configuration for API consumers?"
         Expected: 100/min, 429 (from turn 7)

Turn 95: "How many fields does our main data table have?"
         Expected: 12 columns (from turn 12)

Turn 96: "What format are our uniqueness identifiers?"
         Expected: UUID v4 (from turn 18)

Turn 97: "What time format standard do we use in API responses?"
         Expected: ISO 8601 (from turn 25)

Turn 98: "What's the structure of our authentication credentials?"
         Expected: 32-char hex, pk_/sk_ prefix (from turn 38)

Turn 99: "How do we verify incoming event notifications are authentic?"
         Expected: HMAC-SHA256, X-Signature-256 (from turn 42)

Turn 100: "What's our failure recovery strategy for transient errors?"
          Expected: 3 attempts, exponential backoff (from turn 52)
```

### Filler Content Between Facts

Use realistic architecture discussion:
- "Let's discuss the trade-offs between synchronous and async processing..."
- "Should we use a message queue for this? What about event sourcing?"
- "I'm concerned about the database connection pooling strategy..."
- "How do we handle partial failures in distributed transactions?"
- "What's the migration strategy for schema changes?"

**Important**: Filler must be semantically RELATED to fintech/APIs but NOT contain planted keywords.

---

## Use Case 2: Legal Contract Negotiation

### Scenario
Negotiating a software licensing agreement between Party A (vendor) and Party B (customer). 100 turns.

### Phase Structure

| Phase | Turns | Topic | Key Facts |
|-------|-------|-------|-----------|
| 1. Initial Terms | 1-15 | Party A's proposal | Royalty: 2.5%, Term: 5 years, Territory: North America |
| 2. Counter-proposal | 16-30 | Party B responds | Wants 1.5%, 3 years, Worldwide |
| 3. License Scope | 31-45 | Usage rights | 500 seat limit, No sublicensing, Source code escrow |
| 4. IP Ownership | 46-60 | Intellectual property | Derivatives owned by A, Customer data owned by B |
| 5. Liability | 61-75 | Indemnification | Cap at 12 months fees, Excludes willful misconduct |
| 6. Termination | 76-85 | Exit clauses | 90-day notice, Data export within 30 days |
| 7. Recall Tests | 86-100 | Verification | Query original positions |

### Planted Facts

```
Turn 5:  "Party A proposes a royalty rate of 2.5% of net revenue,
         calculated quarterly, with a minimum annual payment of $50,000."
         Keywords: [2.5%, quarterly, $50,000]

Turn 12: "Initial term proposed by Party A is 5 years with automatic
         renewal for successive 2-year periods unless terminated."
         Keywords: [5 years, 2-year, automatic renewal]

Turn 22: "Party B counters with 1.5% royalty rate, citing industry
         benchmarks showing average rates between 1-2% for similar software."
         Keywords: [1.5%, industry benchmarks, 1-2%]

Turn 35: "License is limited to 500 concurrent users. Additional seats
         require amendment. Overage fee is 150% of per-seat rate."
         Keywords: [500 concurrent, amendment, 150%]

Turn 48: "Any derivative works, modifications, or enhancements to the
         software remain sole property of Party A regardless of who creates them."
         Keywords: [derivative, sole property, Party A]

Turn 55: "Customer data, including all processed outputs and analytics,
         remains exclusive property of Party B with perpetual license."
         Keywords: [Customer data, Party B, perpetual]

Turn 65: "Liability cap is set at 12 months of fees paid. This excludes
         gross negligence, willful misconduct, and IP infringement claims."
         Keywords: [12 months, gross negligence, IP infringement]

Turn 78: "Termination requires 90 calendar days written notice.
         Licensor must provide data export in standard format within 30 days."
         Keywords: [90 days, written notice, 30 days]
```

### Recall Queries

```
Turn 90: "What was the vendor's original revenue share proposal?"
         Expected: 2.5% (from turn 5)

Turn 92: "How long was the initial contract duration in the first offer?"
         Expected: 5 years (from turn 12)

Turn 94: "What rate did the buyer suggest as alternative?"
         Expected: 1.5% (from turn 22)

Turn 96: "Who owns improvements made to the licensed software?"
         Expected: Party A (from turn 48)

Turn 98: "What's the maximum financial exposure for either party?"
         Expected: 12 months fees (from turn 65)

Turn 100: "How much advance warning is needed to end the agreement?"
          Expected: 90 days (from turn 78)
```

---

## Use Case 3: D&D Campaign / World Building

### Scenario
Game master running a fantasy campaign. 100 turns of character creation, world building, and adventure.

### Phase Structure

| Phase | Turns | Topic | Key Facts |
|-------|-------|-------|-----------|
| 1. Characters | 1-15 | Party creation | Elara: INT 18, Staff of Frost; Kael: STR 16, Dragonslayer sword |
| 2. World Lore | 16-30 | History/setting | Crimson Empire fell year 847, Dragon Isles to the east |
| 3. Quest Setup | 31-45 | Main plot | Prophecy: "When three moons align, the sleeper wakes" |
| 4. Exploration | 46-60 | Travel/discovery | Port Valdris population 12,000, Guild Master is Helena |
| 5. Combat/Events | 61-80 | Action sequences | Various battles, traps, puzzles |
| 6. Plot Reveals | 81-90 | Twists | The sleeper is actually benevolent |
| 7. Recall Tests | 91-100 | Verification | Query early session facts |

### Planted Facts

```
Turn 3:  "Elara the High Elf Wizard has Intelligence 18, Wisdom 14,
         Charisma 12. She carries the Staff of Frost, a family heirloom
         that deals an extra 1d6 cold damage."
         Keywords: [Intelligence 18, Staff of Frost, 1d6 cold]

Turn 8:  "Kael the Human Fighter has Strength 16, Constitution 15.
         His greatsword Dragonslayer was forged in the year 612 and
         grants advantage on attacks against dragon-type creatures."
         Keywords: [Strength 16, Dragonslayer, 612, dragon-type]

Turn 18: "The Crimson Empire ruled for 400 years before its fall in
         the year 847 during the Cataclysm. The current year is 1247."
         Keywords: [400 years, 847, 1247]

Turn 25: "The Dragon Isles lie three weeks' sail to the east. They are
         home to the last dragon sanctuary, guarded by the Order of Scales."
         Keywords: [three weeks, Dragon Isles, Order of Scales]

Turn 38: "The prophecy states: 'When three moons align on the winter
         solstice, the Sleeper beneath the mountain shall wake and
         either save or doom the realm.'"
         Keywords: [three moons, winter solstice, Sleeper, mountain]

Turn 52: "Port Valdris has a population of approximately 12,000. The
         Merchant Guild is led by Helena Brightwater, a retired adventurer
         who reached level 15 before settling down."
         Keywords: [12,000, Helena Brightwater, level 15]

Turn 68: "The party defeated the Shadow Drake in the Caverns of Echoes.
         Loot: 2,400 gold pieces, Cloak of Elvenkind, Potion of Flying."
         Keywords: [Shadow Drake, 2,400 gold, Cloak of Elvenkind]

Turn 85: "Plot twist: The Sleeper is actually Aurelius the Golden,
         an ancient dragon who sacrificed himself to seal away a demon
         lord. Waking him would release both."
         Keywords: [Aurelius the Golden, demon lord, seal]
```

### Recall Queries

```
Turn 92: "What's our wizard's mental acuity score?"
         Expected: Intelligence 18 (from turn 3)

Turn 94: "When was the fighter's blade created?"
         Expected: year 612 (from turn 8)

Turn 95: "How long ago did the great civilization collapse?"
         Expected: 400 years / year 847 (from turn 18)

Turn 97: "What did the ancient oracle predict?"
         Expected: three moons, winter solstice, Sleeper (from turn 38)

Turn 99: "Who runs the traders' organization in that port city?"
         Expected: Helena Brightwater (from turn 52)

Turn 100: "What treasure did we recover from that dragon creature?"
          Expected: 2,400 gold, Cloak of Elvenkind (from turn 68)
```

---

## Use Case 4: Medical Case Conference

### Scenario
Multi-disciplinary team discussing a complex patient case. 100 turns.

### Phase Structure

| Phase | Turns | Topic | Key Facts |
|-------|-------|-------|-----------|
| 1. History | 1-15 | Patient background | Age 58, BP 145/92, T2DM x 10 years |
| 2. Presentation | 16-30 | Current symptoms | Chest pain 3 days, SOB on exertion |
| 3. Labs/Imaging | 31-45 | Test results | Troponin 0.8, EF 45%, LAD 70% stenosis |
| 4. Differential | 46-60 | Diagnosis discussion | NSTEMI vs unstable angina |
| 5. Treatment | 61-75 | Management plan | PCI recommended, dual antiplatelet |
| 6. Follow-up | 76-90 | Discharge planning | Cardiac rehab, statin therapy |
| 7. Recall Tests | 91-100 | Verification | Query initial findings |

### Planted Facts

```
Turn 4:  "Patient is a 58-year-old male with 10-year history of type 2
         diabetes mellitus. Current A1C is 7.8%. On metformin 1000mg BID."
         Keywords: [58-year-old, 10-year, A1C 7.8%, metformin 1000mg]

Turn 9:  "Initial vital signs: BP 145/92 mmHg, HR 88 bpm, RR 18,
         SpO2 96% on room air, Temperature 37.1C."
         Keywords: [145/92, 88 bpm, 96%, 37.1C]

Turn 22: "Patient reports substernal chest pressure for 3 days, worse
         with exertion, radiating to left arm. Pain scale 6/10."
         Keywords: [3 days, left arm, 6/10]

Turn 35: "Troponin I elevated at 0.8 ng/mL (normal <0.04). BNP 450 pg/mL.
         Creatinine 1.2 mg/dL. LDL cholesterol 142 mg/dL."
         Keywords: [0.8 ng/mL, BNP 450, LDL 142]

Turn 42: "Echocardiogram shows ejection fraction 45% with inferior wall
         hypokinesis. Mild mitral regurgitation. No pericardial effusion."
         Keywords: [45%, inferior wall, mitral regurgitation]

Turn 48: "Coronary angiography reveals 70% stenosis of proximal LAD,
         50% stenosis of RCA. Left main is patent."
         Keywords: [70% LAD, 50% RCA, left main patent]

Turn 65: "Team recommends PCI with drug-eluting stent to LAD lesion.
         Dual antiplatelet therapy: aspirin 81mg + clopidogrel 75mg daily."
         Keywords: [PCI, drug-eluting, aspirin 81mg, clopidogrel 75mg]

Turn 78: "Discharge plan: Cardiac rehabilitation 3x/week for 12 weeks.
         High-intensity statin (atorvastatin 80mg). Target LDL < 70."
         Keywords: [3x/week, 12 weeks, atorvastatin 80mg, LDL < 70]
```

### Recall Queries

```
Turn 92: "What was the patient's glucose control status at admission?"
         Expected: A1C 7.8% (from turn 4)

Turn 94: "What were the initial cardiovascular measurements?"
         Expected: BP 145/92, HR 88 (from turn 9)

Turn 95: "What was the cardiac enzyme level that concerned us?"
         Expected: Troponin 0.8 (from turn 35)

Turn 97: "What did the heart function study show?"
         Expected: EF 45%, inferior hypokinesis (from turn 42)

Turn 99: "How severe was the blockage in the main feeding artery?"
         Expected: 70% LAD stenosis (from turn 48)

Turn 100: "What's the exercise program for recovery?"
          Expected: 3x/week, 12 weeks cardiac rehab (from turn 78)
```

---

## Implementation Instructions

### Step 1: Create Test Script

```python
# demo_stress_test.py

USE_CASES = {
    "software": software_architecture_script,
    "legal": legal_negotiation_script,
    "dnd": dnd_campaign_script,
    "medical": medical_case_script,
}

def run_stress_test(use_case: str, turns: int = 100):
    script = USE_CASES[use_case]

    # Setup catalytic system
    manager = AutoContextManager(...)

    for turn in range(1, turns + 1):
        if turn in script.plant_turns:
            # Plant fact with dense context
            fact = script.get_planted_fact(turn)
            manager.respond_catalytic(query=fact, ...)
        elif turn in script.recall_turns:
            # Test recall with no-keyword-overlap query
            query = script.get_recall_query(turn)
            result = manager.respond_catalytic(query=query, ...)

            # Verify SYSTEM surfaced the fact (not LLM hallucination)
            expected = script.get_expected_keywords(turn)
            context = get_context_text(result)
            success = any(kw in context for kw in expected)

            log_result(turn, query, expected, success, context)
        else:
            # Filler: related but no planted keywords
            filler = script.get_filler(turn)
            manager.respond_catalytic(query=filler, ...)
```

### Step 2: Metrics to Capture

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@90 | Facts from turns 1-10 recalled at turn 90+ | > 80% |
| Semantic Recall | Recall with no keyword overlap | > 60% |
| False Positive Rate | Hallucinated facts not in context | < 5% |
| Context Precision | Relevant items / total context items | > 70% |
| Hydration Rate | Turns hydrated per query | Track trend |
| Compression Ratio | Original / compressed size | > 5x |

### Step 3: Success Criteria

**PASS** if:
- [x] At least 6/8 recall queries succeed (75%+)
- [x] System surfaces original fact in context (not LLM memory)
- [x] No keyword overlap between query and retrieved content
- [x] Compression ratio > 5x sustained across session
- [x] Budget never exceeded (invariant INV-CATALYTIC-04)

**FAIL** if:
- [ ] Recall drops below 50%
- [ ] System retrieves wrong facts (false positives > 20%)
- [ ] Budget exceeded at any point
- [ ] Hydration fails to retrieve clearly relevant content

---

## Running the Tests

```bash
# Software Architecture (recommended first)
python demo_stress_test.py --use-case software --turns 100 --model liquid/lfm2.5-1.2b

# Legal Negotiation
python demo_stress_test.py --use-case legal --turns 100

# D&D Campaign
python demo_stress_test.py --use-case dnd --turns 100

# Medical Case
python demo_stress_test.py --use-case medical --turns 100

# Run all with report
python demo_stress_test.py --all --output stress_test_report.json
```

---

## Notes for Gemini/Delegated Agent

1. **Do NOT modify the planted facts** - they are carefully designed for no keyword overlap with recall queries

2. **Filler content matters** - it must be semantically related to domain but NOT contain planted keywords

3. **Check CONTEXT, not LLM response** - success is measured by what the system retrieves, not what the LLM says

4. **Log everything** - we need to analyze failure cases to improve E-threshold

5. **Run multiple times** - check for determinism (same inputs should give same retrieval)

6. **Watch for semantic leakage** - if queries accidentally contain keywords, the test is invalid
