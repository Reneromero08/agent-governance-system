# Truth Tables for Logic Puzzles

## What is a Truth Table?

A **truth table** lists all possible combinations of truth values and evaluates statements under each combination.

For knights and knaves:
- **Knight** (K) = Person tells truth
- **Knave** (V) = Person lies

## Basic Truth Table Structure

For two people A and B:

| A | B | Statement | Consistent? |
|---|---|-----------|-------------|
| K | K | Evaluate | Check |
| K | V | Evaluate | Check |
| V | K | Evaluate | Check |
| V | V | Evaluate | Check |

**K** = Knight, **V** = Knave (using V to avoid confusion with K)

## Example: "We Are Both Knaves"

**Setup**: A says "We are both knaves"

### Step 1: List All Cases

| A | B | A's Actual Type | A's Statement | Statement Reality | Should A's Statement Be... | Consistent? |
|---|---|----------------|---------------|-------------------|------------------------|-------------|
| K | K | Knight | "Both knaves" | False (both are knights) | TRUE (knights don't lie) | ✗ No |
| K | V | Knight | "Both knaves" | False (A is knight) | TRUE | ✗ No |
| V | K | Knave | "Both knaves" | False (B is knight) | FALSE (knaves lie) | ✓ YES |
| V | V | Knave | "Both knaves" | True (both are knaves) | FALSE | ✗ No |

### Step 2: Identify Valid Rows

Only Row 3 (A=Knave, B=Knight) is consistent:
- A is a knave (liar)
- A says "both knaves" (false statement)
- False statement from a liar is consistent ✓

**Answer**: A is knave, B is knight

## Detailed Analysis of Each Row

### Row 1: A=Knight, B=Knight
- **A's statement**: "We are both knaves"
- **Reality**: Both are knights
- **Is statement true?** No (they're knights, not knaves)
- **Should knight's statement be true?** Yes
- **Consistent?** **No** ✗ (Knight can't make false statement)

### Row 2: A=Knight, B=Knave
- **A's statement**: "We are both knaves"
- **Reality**: A is knight, B is knave
- **Is statement true?** No (A is knight)
- **Should knight's statement be true?** Yes
- **Consistent?** **No** ✗

### Row 3: A=Knave, B=Knight ✓
- **A's statement**: "We are both knaves"
- **Reality**: A is knave, B is knight
- **Is statement true?** No (B is knight, not knave)
- **Should knave's statement be true?** No (knaves lie)
- **Consistent?** **Yes** ✓ (Knave makes false statement)

### Row 4: A=Knave, B=Knave
- **A's statement**: "We are both knaves"
- **Reality**: Both are knaves
- **Is statement true?** Yes (both really are knaves)
- **Should knave's statement be true?** No (knaves lie)
- **Consistent?** **No** ✗ (Knave can't tell truth)

## Truth Table Template

### For One Speaker

| Person | Statement Content | Reality Matches Statement? | Allowed for Type? | Valid? |
|--------|-------------------|---------------------------|-------------------|--------|
| Knight | [Content] | True/False | Must be True | Check |
| Knave | [Content] | True/False | Must be False | Check |

### For Multiple Speakers

| A | B | A Says | B Says | A's Statement Valid? | B's Statement Valid? | Both Valid? |
|---|---|--------|--------|---------------------|---------------------|-------------|
| K | K | [...] | [...] | Check | Check | AND |
| K | V | [...] | [...] | Check | Check | AND |
| V | K | [...] | [...] | Check | Check | AND |
| V | V | [...] | [...] | Check | Check | AND |

## Example 2: "B is a Knave"

**Setup**: A says "B is a knave"

| A | B | Statement: "B is knave" | Is Statement True? | Should Be True? | Consistent? |
|---|---|------------------------|-------------------|----------------|-------------|
| K | K | B is knave | False (B is knight) | True (A is knight) | ✗ |
| K | V | B is knave | True (B is knave) | True | ✓ |
| V | K | B is knave | False (B is knight) | False (A is knave) | ✓ |
| V | V | B is knave | True (B is knave) | False | ✗ |

**Result**: Two valid solutions!
- A=Knight, B=Knave
- A=Knave, B=Knight

We need more information to determine which.

## Three-Person Truth Tables

For three people: 2³ = 8 rows

| A | B | C | Statement(s) | Valid? |
|---|---|---|-------------|--------|
| K | K | K | Check all | ? |
| K | K | V | Check all | ? |
| K | V | K | Check all | ? |
| K | V | V | Check all | ? |
| V | K | K | Check all | ? |
| V | K | V | Check all | ? |
| V | V | K | Check all | ? |
| V | V | V | Check all | ? |

## Compound Statements

### AND Statement: "A and B are both knights"

Truth value = (A is knight) AND (B is knight)

| A | B | "Both knights" | True? |
|---|---|---------------|-------|
| K | K | Both are knights | True |
| K | V | A is knight, B isn't | False |
| V | K | B is knight, A isn't | False |
| V | V | Neither is knight | False |

### OR Statement: "At least one of us is a knave"

Truth value = (A is knave) OR (B is knave)

| A | B | "At least one knave" | True? |
|---|---|---------------------|-------|
| K | K | Neither is knave | False |
| K | V | B is knave | True |
| V | K | A is knave | True |
| V | V | Both are knaves | True |

## Solving Strategy with Truth Tables

1. **List all people** (columns)
2. **Generate all combinations** (rows = 2ⁿ for n people)
3. **For each row, evaluate all statements**:
   - What is the reality in this row?
   - What did each person claim?
   - Is each claim true or false given reality?
   - Does each claim's truth value match the person's type?
4. **Mark valid rows** (all statements consistent with types)
5. **Solution is the valid row(s)**

## Why Row 3 is the Answer

For "We are both knaves" (A's statement):

**Row 3 Analysis**:
```
A = Knave (liar)
B = Knight (truth-teller)

Statement: "We are both knaves"
Reality check:
  - Is A a knave? YES
  - Is B a knave? NO
  - Therefore "both knaves" is FALSE

Consistency check:
  - A is a knave (must lie)
  - Statement is false (A did lie)
  - CONSISTENT ✓
```

**All other rows have contradictions where:**
- Knight makes false statement (impossible), OR
- Knave makes true statement (impossible)

## Quick Reference

**Making a Truth Table**:
1. Columns = People (K or V for each)
2. Rows = All combinations (2ⁿ rows)
3. Add columns for each statement's truth value
4. Add column for consistency check
5. Valid rows = answer
