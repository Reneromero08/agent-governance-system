# Logical Deduction Techniques

## What is Logical Deduction?

**Deduction**: Drawing necessary conclusions from given premises using logical rules.

If premises are true and logic is valid, conclusion MUST be true.

## Basic Logical Forms

### Modus Ponens (Affirming the Antecedent)

**Form**:
1. If P, then Q
2. P is true
3. Therefore, Q is true

**Example**:
1. If it's raining, the ground is wet
2. It's raining
3. Therefore, the ground is wet ✓

### Modus Tollens (Denying the Consequent)

**Form**:
1. If P, then Q
2. Q is false
3. Therefore, P is false

**Example**:
1. If A is a knight, A's statements are true
2. A's statement is false
3. Therefore, A is not a knight (A is a knave) ✓

This is crucial for knights and knaves!

### Disjunctive Syllogism

**Form**:
1. P or Q (at least one is true)
2. Not P
3. Therefore, Q

**Example**:
1. A is a knight or a knave
2. A is not a knight
3. Therefore, A is a knave ✓

## Contradiction Method (Reductio ad Absurdum)

**Strategy**: Assume something, derive a contradiction, conclude the assumption was false.

### Example: Proving A is a Knave

**Given**: A says "We are both knaves"

**Step 1**: Assume A is a knight
- If A is knight, A's statements are true
- A says "we're both knaves"
- So A is a knave (from the true statement)
- But we assumed A is a knight!
- **Contradiction**: A cannot be both knight and knave

**Step 2**: Conclusion
- Since assuming "A is knight" leads to contradiction
- A must NOT be a knight
- Since everyone is knight or knave
- **Therefore A is a knave**

## Case Analysis (Exhaustive Cases)

When there are limited possibilities, check each:

### Two-Person Problem

**Possible cases**:
1. A=Knight, B=Knight
2. A=Knight, B=Knave
3. A=Knave, B=Knight
4. A=Knave, B=Knave

**Strategy**: Test each case against all statements
- Eliminate cases with contradictions
- Remaining case(s) are the solution

### Example: A Says "At Least One of Us is a Knave"

**Case 1: A=Knight, B=Knight**
- A is knight → statement must be true
- Statement: "At least one is knave"
- Reality: Both are knights (no knaves)
- **Contradiction** ✗

**Case 2: A=Knight, B=Knave**
- A is knight → statement must be true
- Statement: "At least one is knave"
- Reality: B is knave (at least one knave exists)
- **Consistent** ✓

**Case 3: A=Knave, B=Knight**
- A is knave → statement must be false
- Statement: "At least one is knave"
- Reality: A is knave (at least one exists)
- For statement to be false, no knaves should exist
- **Contradiction** ✗

**Case 4: A=Knave, B=Knave**
- A is knave → statement must be false
- Statement: "At least one is knave"
- Reality: Both are knaves (at least one exists)
- **Contradiction** ✗

**Solution**: Only Case 2 works → A is knight, B is knave

## Implication Chains

Link multiple implications:

**Example**:
1. If A is knight → A's statement is true
2. If A's statement is true → B is knave (from A's claim)
3. Therefore: If A is knight → B is knave

**Contrapositive**: If B is not knave → A is not knight

## Solving "We Are Both Knaves"

Let's apply these techniques systematically:

**Given**: A says "We are both knaves"

### Method 1: Contradiction

**Assume A is knight**:
- Knights tell truth
- So "we're both knaves" is true
- So A is a knave
- But A is a knight (assumption)
- **Contradiction!**

**Conclude**: A is NOT a knight → A is knave

**Determine B**:
- A is knave, so A's statement is false
- Statement: "Both are knaves"
- False statement means: NOT both are knaves
- We know A is knave
- So B must NOT be knave
- **Therefore B is knight**

### Method 2: Case Analysis

**Case 1: A=Knight, B=Knight**
- A claims "both knaves"
- Reality: Both knights
- Knights tell truth, but statement is false
- **Contradiction** ✗

**Case 2: A=Knight, B=Knave**
- A claims "both knaves"
- Reality: A is knight, B is knave
- Knights tell truth, but statement is false (A isn't knave)
- **Contradiction** ✗

**Case 3: A=Knave, B=Knight**
- A claims "both knaves"
- Reality: A is knave, B is knight
- Knaves lie, statement is false (B isn't knave)
- Statement is indeed false ✓
- **Consistent** ✓

**Case 4: A=Knave, B=Knave**
- A claims "both knaves"
- Reality: Both are knaves
- Knaves lie, but statement is TRUE
- **Contradiction** ✗

**Solution**: Only Case 3 → A is knave, B is knight

## Key Principles

1. **Excluded Middle**: Everyone is either knight or knave (no third option)
2. **Non-Contradiction**: Can't be both knight and knave simultaneously
3. **Consistency**: All statements must match the person's type
4. **Falsity Check**: For knaves, ask "Is this statement false given reality?"

## Practice Strategy

1. List all statements
2. List all people
3. Write out cases (2ⁿ cases for n people)
4. For each case, check every statement
5. Eliminate contradictory cases
6. Verify remaining solution(s)

## Common Mistakes to Avoid

❌ **"A says X, so X is true"** - Wrong! Depends if A is knight or knave

❌ **"B doesn't speak, so B could be anything"** - True, but A's statements might constrain B

❌ **Forgetting to check all statements** - Every statement must be consistent

✓ **Check BOTH "knight assumption" AND "knave assumption"**

✓ **Verify final answer against all statements**
