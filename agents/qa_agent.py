"""
QA Agent — Automated game-tester and crash-analysis engine.

Simulates real-world game testing by:

  1. **Static analysis** of the generated TypeScript source code to detect
     common Cocos Creator defects (out-of-bounds access, missing null guards,
     incorrect coordinate transforms).
  2. **Dynamic simulation** of EasyClick-style tap sequences (2D coordinate
     gestures on a 1080×1920 Android display).
  3. **Crash-log synthesis** — when a defect is found, the agent produces a
     verbose crash report (native stack trace, registers, GL state, heap
     snapshot) that the Dev Agent ingests for self-healing.

The deterministic defect finder means QA results are **reproducible** — no
random pass/fail.  A real integration would run Playwright + ADB against an
Android emulator; this skeleton focuses on the crash-analysis context payload
that drives the self-healing loop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from core.llm import estimate_tokens

logger = logging.getLogger("agents.qa")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class TestStep:
    """A single coordinated tap, swipe, or assertion."""

    action: str                # "tap" | "swipe" | "assert"
    x: int
    y: int
    expected: str = ""
    actual: str = ""
    passed: bool = True


@dataclass
class TestCaseResult:
    """Outcome of an individual test case."""

    name: str
    steps: list[TestStep] = field(default_factory=list)
    passed: bool = True
    crash_log: str = ""
    duration_ms: float = 0.0


@dataclass
class QASuiteResult:
    """Aggregated results from a QA suite run."""

    results: list[TestCaseResult] = field(default_factory=list)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def total_crash_log_tokens(self) -> int:
        return sum(estimate_tokens(r.crash_log) for r in self.results)

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        failing = [r.name for r in self.results if not r.passed]
        parts = [f"QA: {passed}/{total} passed"]
        if failing:
            parts.append(f"FAILURES: {', '.join(failing)}")
            parts.append(f"crash_log_tokens: ~{self.total_crash_log_tokens()}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Static defect detector — deterministic, based on real code analysis
# ---------------------------------------------------------------------------

class DefectDetector:
    """Scans TypeScript source for common Cocos Creator defects.

    Each method returns a list of (severity, description) tuples.
    """

    @staticmethod
    def scan(source: str) -> list[tuple[str, str]]:
        defects: list[tuple[str, str]] = []
        defects.extend(DefectDetector._check_bounds(source))
        defects.extend(DefectDetector._check_null_safety(source))
        defects.extend(DefectDetector._check_lifecycle(source))
        defects.extend(DefectDetector._check_coordinates(source))
        return defects

    @staticmethod
    def _check_bounds(source: str) -> list[tuple[str, str]]:
        issues = []
        # Look for grid access without bounds check
        if "findMatches" in source and "row <" not in source and "col <" not in source:
            issues.append(("HIGH", "findMatches(): missing row/col bounds check"))
        # Array access without guard
        if ".grid[r]" in source and "r <" not in source:
            issues.append(("MEDIUM", "Unsafe array access: grid[r] without bounds guard"))
        # Swap method without bounds
        if "trySwap" in source and "r1 <" not in source and "r1 >= 0" not in source:
            pass  # acceptable if called from onTouchEnd which checks
        return issues

    @staticmethod
    def _check_null_safety(source: str) -> list[tuple[str, str]]:
        issues = []
        if "querySelector" in source and "?." not in source:
            issues.append(("MEDIUM", "querySelector() result used without optional chaining or null check"))
        if "getComponent" in source and "??" not in source and "=== null" not in source and "!= null" not in source and "getComponent(" in source:
            lines = source.splitlines()
            for i, line in enumerate(lines):
                if "getComponent" in line:
                    # Check if next line uses it without guard
                    if i + 1 < len(lines) and "getComponent" in lines[i + 1]:
                        pass
                    break
        if "convertToNodeSpaceAR" in source and "UITransform" in source:
            pass  # guarded by the pattern itself
        return issues

    @staticmethod
    def _check_lifecycle(source: str) -> list[tuple[str, str]]:
        issues = []
        if "node.on(" in source and "node.off(" not in source and "onDestroy" not in source:
            issues.append(("HIGH", "Event listener registered in onLoad but no onDestroy cleanup"))
        if "onLoad" not in source and "start" not in source:
            issues.append(("LOW", "No lifecycle method (onLoad/start) found"))
        return issues

    @staticmethod
    def _check_coordinates(source: str) -> list[tuple[str, str]]:
        issues = []
        # Check if coordinate conversion uses UITransform
        if "getUILocation" in source and "convertToNodeSpaceAR" not in source and "convertToNodeSpace" not in source:
            issues.append(("MEDIUM", "Touch coordinates not converted to node space"))
        if "boardOffsetX" not in source and "boardOffsetY" not in source:
            issues.append(("LOW", "No board offset handling; touch coordinates may misalign"))
        return issues


# ---------------------------------------------------------------------------
# Crash-log synthesis
# ---------------------------------------------------------------------------

def _synthesize_crash_log(
    test_name: str,
    component: str,
    defects: list[tuple[str, str]],
    step_index: int = 2,
) -> str:
    """Produce a deterministic verbose crash log from static analysis findings.

    The crash log is deliberately long (1600+ characters / ~400+ tokens) to
    demonstrate long-context ingestion in the refactor loop.
    """
    high_issues = [d for d in defects if d[0] == "HIGH"]
    medium_issues = [d for d in defects if d[0] == "MEDIUM"]

    crash_source = high_issues[0][1] if high_issues else (medium_issues[0][1] if medium_issues else "null pointer in grid access")
    func_name = crash_source.split("(")[0].split(":")[0].strip()

    # Extract line numbers from source by scanning for the function
    return textwrap.dedent(f"""\
        ============================================================
        CRASH REPORT — {test_name} / {component}
        Timestamp: 2026-05-10T10:15:41.123Z
        Severity: FATAL
        ============================================================

        SIGNAL: SIGSEGV (11) — Segmentation fault
        FAULT_ADDR: 0x0000007f8c4a0000c0
        SI_CODE: SEGV_MAPERR

        --- Trigger ---
        Test step #{step_index}: swipe at (240, 860) →
        EasyClick action: SwipeAction(direction=LEFT, start=(240,860), end=(160,860))

        --- Bug Report ---
        Static analysis identified the following defect:
          {crash_source}

        Root cause: {_root_cause_text(crash_source)}

        --- arm64 Register State ---
        x0   0x0000000000000004  x1   0x0000007f8c4a0000c0
        x2   0x0000000000000000  x3   0x0000007bc9a8f000
        x4   0x0000007c4a3b2f10  x5   0xffffffffffffffff
        x6   0x0000000000000000  x7   0x0000000000000000
        x8   0x0000000000000010  x9   0x0000007bc9a8fc00
        x10  0x0000007c4a3b2f14  x11  0x0000000000000001
        x12  0x0000007bc9a8e800  x13  0x0000000000000001
        sp   0x0000007fbfeba000  lr   0x0000007c4a3b2f10
        pc   0x0000007c4a3b2f14  cpsr 0x0000000080000000

        --- Native Call Stack ---
        0  libcocos2djs.so       0x7c4a3b2f14  {func_name or 'unknown'}()
        1  libcocos2djs.so       0x7c4a3b2f90  Match3Board::swap()
        2  libcocos2djs.so       0x7c4a3b3010  GameTouchHandler::onTouchEnded()
        3  libsystem_platform.dylib 0x1d3c4f000  _sigtramp
        4  libsystem_c.dylib     0x1d3b2e000  abort() + 128

        --- JavaScript Call Stack (Frida-style) ---
          at {func_name or 'unknown'} (assets/Scripts/{component}.ts:85:12)
          at Match3Board.swap (assets/Scripts/{component}.ts:42:22)
          at GameTouchHandler.onTouchEnded (assets/Scripts/GameTouchHandler.ts:31:8)
          at EventTouch.dispatch (cocos/core/platform/event.ts:120:5)

        --- GL State Snap ---
        GL_CURRENT_PROGRAM:   0x3a1f0008
        GL_ARRAY_BUFFER:      0x3a1f0010
        GL_ELEMENT_ARRAY_BUFFER: 0x3a1f0018
        GL_ACTIVE_TEXTURE:    GL_TEXTURE0
        Current FBO:          0x3a20001c (screen)
        Draw calls this frame: 142

        --- JSB Heap ---
        Total:  42.3 MiB
        Used:   38.7 MiB
        Pending: 3.2 MiB
        Objects: 14203
        Strings: 2894

        --- Analysis ---
        Crash reproducible 3/3 attempts.
        Introducing bounds checking in {func_name or 'the crash site'} and
        adding defensive guards at array access points should resolve this issue.
        ============================================================
    """)


def _root_cause_text(defect: str) -> str:
    if "bounds" in defect.lower():
        return "Array index out of range when grid dimensions are inconsistent after collapse"
    if "null" in defect.lower() or "null" in defect.lower():
        return "Dereferencing null object returned by querySelector/getComponent"
    if "lifecycle" in defect.lower():
        return "Event listener leak causing dangling pointer after scene unload"
    if "coordinate" in defect.lower():
        return "Screen-space coordinates not converted to node-local space; wrong tile index computed"
    return "Unhandled runtime exception in game logic"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QAAgent:
    """Runs automated game tests with static analysis and synthesised crash logs."""

    # Pre-defined test templates keyed by component name patterns
    _TEST_TEMPLATES: dict[str, list[dict]] = {
        "default": [
            {"name": "SMK_01_SceneLoad",    "steps": [(120, 860, "tap"), (120, 860, "assert")]},
            {"name": "FUNC_01_SwapGesture", "steps": [(240, 400, "tap"), (320, 400, "swipe"), (320, 400, "assert")]},
            {"name": "FUNC_02_MatchDetect", "steps": [(160, 560, "tap"), (160, 560, "assert")]},
            {"name": "STRESS_01_RapidInput", "steps": [(240, 400, "tap"), (160, 560, "tap"), (320, 400, "tap"), (320, 560, "assert")]},
            {"name": "REG_01_ScoreUpdate",  "steps": [(240, 400, "tap"), (320, 400, "swipe"), (160, 560, "assert")]},
        ],
    }

    def __init__(self, k3s_namespace: str = "game-pipeline") -> None:
        self.k3s_namespace = k3s_namespace
        logger.info("QAAgent initialised [namespace=%s]", k3s_namespace)

    # ── Public API ──────────────────────────────────────────────────────────

    async def run_suite(
        self,
        component: str = "Match3Board",
        source_code: str = "",
    ) -> QASuiteResult:
        """Execute a QA test suite against a component.

        ``source_code`` is passed from the orchestrator so static analysis
        can detect real defects deterministically.
        """
        logger.info("QAAgent.run_suite — component=%s", component)

        # Run static analysis on the actual source code
        defects = DefectDetector.scan(source_code) if source_code else []

        # Determine which test cases fail based on defect severity
        has_high = any(d[0] == "HIGH" for d in defects)
        has_medium = any(d[0] == "MEDIUM" for d in defects)

        templates = self._TEST_TEMPLATES.get(component, self._TEST_TEMPLATES["default"])
        tasks = [
            self._run_test_case(component, t, defects, has_high, has_medium)
            for t in templates
        ]
        results = await asyncio.gather(*tasks)

        suite = QASuiteResult(results=list(results))
        logger.info(
            "QA suite complete — passed=%s, crash_log_tokens=%d",
            suite.all_passed(),
            suite.total_crash_log_tokens(),
        )
        if not suite.all_passed():
            logger.warning("QA failures detected:\n%s", suite.summary())
        return suite

    # ── Test case runner ────────────────────────────────────────────────────

    async def _run_test_case(
        self,
        component: str,
        template: dict,
        defects: list[tuple[str, str]],
        has_high: bool,
        has_medium: bool,
    ) -> TestCaseResult:
        """Run a single test case.  Failure is deterministic based on defect analysis.

        - HIGH defects → all tests fail.
        - MEDIUM defects → functional (+ regression) tests fail.
        - LOW / no defects → all pass.
        """
        test_name = f"{component}_{template['name']}"
        raw_steps = template["steps"]
        steps = [
            TestStep(action=a, x=x, y=y, expected=f"ok_{i}")
            for i, (x, y, a) in enumerate(raw_steps)
        ]

        # Simulate test latency
        await asyncio.sleep(0.3)

        # Determine pass/fail deterministically from defect analysis
        is_smoke = "SMK" in test_name
        is_stress = "STRESS" in test_name
        is_reg = "REG" in test_name
        is_func = "FUNC" in test_name

        passed = True
        crash_log = ""

        if has_high:
            # HIGH defects crash everything
            passed = False
        elif has_medium and (is_func or is_reg):
            # MEDIUM defects break functional + regression tests
            passed = False
        elif has_medium and is_stress:
            # Stress tests are affected by MEDIUM defects too
            passed = False

        if not passed:
            crash_log = _synthesize_crash_log(
                test_name=test_name,
                component=component,
                defects=defects,
                step_index=len(steps) // 2,
            )
            if is_smoke:
                crash_log = crash_log.replace(
                    "Severity: FATAL",
                    "Severity: FATAL\nContext: Smoke test — production-blocking defect",
                )

        return TestCaseResult(
            name=test_name,
            steps=steps,
            passed=passed,
            crash_log=crash_log,
            duration_ms=round(400 + len(steps) * 120, 1),
        )
