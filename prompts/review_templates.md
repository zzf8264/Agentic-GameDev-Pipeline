# Code Review Templates — Agentic GameDev Pipeline
#
# These templates are used by the Dev Agent during the self-correction loop
# and by the Orchestrator when evaluating generated code before passing it
# to QA.  Each template is designed to produce consistent, actionable
# review comments.

---

## Template 1 — Standard Code Review

**Reviewer:** DevAgent (self-review pass)

### Checklist
- [ ] Code compiles without TypeScript errors.
- [ ] All `@ccclass` decorators are present.
- [ ] Touch handlers use 2D coordinates (x, y).
- [ ] No magic numbers — grid dimensions, tile sizes are `@property` fields.
- [ ] `findMatches()` handles edge cases (empty grid, single row/column).
- [ ] Public API is documented with JSDoc where non-obvious.
- [ ] No `any` types unless absolutely necessary.
- [ ] Memory: no detached event listeners in `onDestroy()`.

### Output Format
```
REVIEW: <component_name>
STATUS: PASS | FAIL | NEEDS_WORK
ISSUES:
  - <severity>: <description> (line: <line_number>)
COMMENTS:
  - <suggestion>
```

---

## Template 2 — Crash-Driven Refactor Review

**Trigger:** QA Agent reported a crash with error log.

**Context provided:**
- Full crash log (registers, stack trace, GL state).
- Original source code.
- Previous review comments (if any).

### Refactor Checklist
- [ ] Identify the root cause from the crash log.
- [ ] Locate the exact line / function in source.
- [ ] Apply minimal fix (avoid scope creep).
- [ ] Verify fix doesn't break existing tests.
- [ ] Add defensive guard (bounds check, null check) as needed.

### Output Format
```
REFACTOR_REVIEW: <component_name>
CRASH_SIGNATURE: <signal> at <address>
ROOT_CAUSE: <one-line explanation>
FIX: <what changed and why>
RISK: LOW | MEDIUM | HIGH
```

---

## Template 3 — Cross-Agent Consistency Check

**When:** After the full Scout → Dev → QA cycle completes.

**Checks:**
- [ ] Does the generated code implement the mechanic identified by Scout?
- [ ] Is the coordinate system consistent across all agents?
- [ ] Do QA crash logs reference functions that exist in the generated code?
- [ ] Has the error-log token count changed between iterations?

### Output Format
```
CROSS_CHECK: <cycle_id>
SCOUT_MECHANIC: <mechanic>
DEV_IMPLEMENTATION: <file> — <status>
QA_RESULT: <pass/fail> — <crash_count>
CONSISTENCY: OK | ISSUES
```

---

## Template 4 — Performance & Token Budget Review

**When:** Before submitting the final API-token quota request.

**Checks:**
- [ ] Estimated total tokens consumed in this session.
- [ ] Number of LLM calls made.
- [ ] Max context window used (largest single prompt).
- [ ] Number of self-healing iterations.
- [ ] Crash-log token contribution to total context.

### Output Format
```
TOKEN_BUDGET: <cycle_id>
TOTAL_LLM_CALLS: <count>
ESTIMATED_TOKENS_IN: <input_tokens>
ESTIMATED_TOKENS_OUT: <output_tokens>
MAX_CONTEXT_WINDOW: <tokens>
SELF_HEALING_LOOPS: <count>
CRASH_LOG_CONTRIBUTION: <tokens> (<percentage>%)
```
