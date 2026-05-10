#!/usr/bin/env python3
"""
Agentic GameDev Pipeline — Orchestrator
========================================

Coordinates the full multi-agent workflow:

    Scout ──► Dev (generate) ──► Dev (review) ──► QA ──► (fail?) ──► Dev (refactor) ──► QA ...

The orchestrator owns:
  - The shared ``LLMClient`` (single OpenAI-compatible connection reused across agents).
  - Pipeline metrics tracking (real token counts from LLM responses).
  - Self-healing loop with configurable iteration budget.
  - Checkpoint persistance (every phase result is saved to ``data/checkpoints/``).
  - Machine-readable metrics output (JSON) for API-token quota applications.

Usage:
    python orchestrator.py                          # Default run
    LLM_ENDPOINT=http://localhost:11434/v1 python orchestrator.py  # Local LLM
    DRY_RUN=true python orchestrator.py              # Skip real LLM calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents import ScoutAgent, DevAgent, QAAgent
from agents.dev_agent import CodeGenRequest, GeneratedCode
from core.llm import LLMClient, estimate_tokens


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    dry_run: bool = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")
    max_self_heal_iterations: int = int(os.getenv("MAX_SELF_HEAL_ITERATIONS", "3"))
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "128000"))
    k3s_namespace: str = os.getenv("K3S_NAMESPACE", "game-pipeline")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "data/checkpoints")
    scout_platform: str = os.getenv("SCOUT_PLATFORM", "douyin")
    scout_hashtags: list[str] = field(
        default_factory=lambda: [
            h.strip() for h in os.getenv("SCOUT_HASHTAGS", "超解压三消,益智游戏开发,cocoscreator").split(",")
        ]
    )
    mechanic: str = os.getenv("GAME_MECHANIC", "match-3")


# ---------------------------------------------------------------------------
# Pipeline metrics
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    run_id: str = ""
    total_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    largest_context_window: int = 0
    self_heal_iterations: int = 0
    crash_logs_processed: int = 0
    crash_log_tokens: int = 0
    scout_trends_found: int = 0
    generated_file: str = ""
    qa_summary: str = ""
    elapsed_seconds: float = 0.0
    phases_completed: list[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_llm_calls": self.total_llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "largest_context_window_tokens": self.largest_context_window,
            "self_heal_iterations": self.self_heal_iterations,
            "crash_logs_processed": self.crash_logs_processed,
            "crash_log_tokens": self.crash_log_tokens,
            "scout_trends_found": self.scout_trends_found,
            "generated_file": self.generated_file,
            "qa_summary": self.qa_summary,
            "phases_completed": list(self.phases_completed),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Drives the Scout → Dev → QA self-healing pipeline."""

    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.metrics = PipelineMetrics(
            run_id=datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
        )
        self._start_time = time.time()

        # Shared LLM client — all agents reuse this connection pool
        self.llm = LLMClient(
            model=self.cfg.llm_model,
            endpoint=self.cfg.llm_endpoint,
            max_tokens=self.cfg.llm_max_tokens,
            dry_run=self.cfg.dry_run,
        )

        # Agents
        self.scout = ScoutAgent(llm=self.llm)
        self.dev = DevAgent(llm=self.llm)
        self.qa = QAAgent(k3s_namespace=self.cfg.k3s_namespace)

        # Checkpoint directory
        self._ckpt_dir = Path(self.cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.log = logging.getLogger("orchestrator")

    # ── Main entry ──────────────────────────────────────────────────────────

    async def run(self) -> PipelineMetrics:
        self.log.info("=" * 72)
        self.log.info("Agentic GameDev Pipeline — %s", self.metrics.run_id)
        self.log.info("Config: model=%s, endpoint=%s, dry_run=%s, max_heal=%d",
                       self.cfg.llm_model, self.cfg.llm_endpoint,
                       self.cfg.dry_run, self.cfg.max_self_heal_iterations)
        self.log.info("=" * 72)

        try:
            # ── Phase 1: Scout recon ─────────────────────────────────
            report = await self._phase_scout()
            self.metrics.phases_completed.append("scout")

            # ── Phase 2: Code generation ─────────────────────────────
            code = await self._phase_generate(report)
            self.metrics.phases_completed.append("generate")

            # ── Phase 3: Self-review ─────────────────────────────────
            code = await self._phase_review(code)
            self.metrics.phases_completed.append("review")

            # ── Phase 4: QA + self-heal loop ─────────────────────────
            code = await self._phase_self_heal(code)
            self.metrics.phases_completed.append("self_heal")

            self.log.info("All phases complete ✓")
        except Exception:
            self.log.exception("Pipeline aborted with exception")
        finally:
            await self.llm.aclose()

        self._finalise_metrics()
        self._log_report()
        self._write_checkpoint("_final_metrics.json", self.metrics.snapshot())
        return self.metrics

    # ── Phase implementations ──────────────────────────────────────────────

    async def _phase_scout(self):
        self.log.info("── Phase 1/4: Scout Reconnaissance ──")
        self.log.info("Platform: %s, hashtags: %s", self.cfg.scout_platform, self.cfg.scout_hashtags)

        report = await self.scout.recon(
            platform=self.cfg.scout_platform,
            hashtags=self.cfg.scout_hashtags,
        )

        self.metrics.scout_trends_found = len(report.trends)
        self.metrics.largest_context_window = max(
            self.metrics.largest_context_window,
            report.total_tokens(),
        )

        self.log.info("Scout found %d trends (~%d tokens)", len(report.trends), report.total_tokens())
        if report.trends:
            self.log.info("Top trend: %s", report.trends[0].short_summary())

        self._write_checkpoint("01_scout_report.json", {
            "trend_count": len(report.trends),
            "context_block": report.to_llm_context_block(),
        })
        return report

    async def _phase_generate(self, report) -> GeneratedCode:
        self.log.info("── Phase 2/4: Code Generation ──")

        context_block = report.to_llm_context_block()

        request = CodeGenRequest(
            trend_context=context_block,
            mechanic=self.cfg.mechanic,
            target_framework="Cocos Creator 3.x",
            style_guide=(
                "Use 2D screen-space coordinates (720×1280, bottom-left origin). "
                "Include swipe-to-match touch handler with UITransform.convertToNodeSpaceAR. "
                "Add @ccclass decorator and export the component."
            ),
        )

        code = await self.dev.generate(request)

        self.metrics.generated_file = code.file_name
        self.metrics.largest_context_window = max(
            self.metrics.largest_context_window,
            estimate_tokens(context_block),
        )

        self.log.info("Generated %s (~%d tokens, %d lines)",
                       code.file_name, code.token_estimate(),
                       len(code.source_code.splitlines()))

        self._write_checkpoint(f"02_gen_{code.file_name.replace('.ts', '.txt')}", code.source_code)
        return code

    async def _phase_review(self, code: GeneratedCode) -> GeneratedCode:
        self.log.info("── Phase 3/4: Code Review ──")

        code = await self.dev.review(code)

        # Track the review LLM call tokens
        self.metrics.total_llm_calls = self.llm.call_count
        self.metrics.prompt_tokens = self.llm.total_prompt_tokens
        self.metrics.completion_tokens = self.llm.total_completion_tokens
        self.metrics.largest_context_window = max(
            self.metrics.largest_context_window,
            estimate_tokens(code.review_comment),
        )

        self.log.info("Review: passed=%s", code.passed_review)
        first_line = code.review_comment.split("\n")[0] if code.review_comment else "(empty)"
        self.log.debug("Review output: %s", first_line)

        self._write_checkpoint("03_review.txt", code.review_comment)
        return code

    async def _phase_self_heal(self, code: GeneratedCode) -> GeneratedCode:
        self.log.info("── Phase 4/4: Self-Healing Loop ──")

        for iteration in range(1, self.cfg.max_self_heal_iterations + 1):
            self.log.info("Iteration %d/%d", iteration, self.cfg.max_self_heal_iterations)

            # ── QA ───────────────────────────────────────────────────
            suite = await self.qa.run_suite(
                component=code.file_name.replace(".ts", ""),
                source_code=code.source_code,
            )

            crash_tokens = suite.total_crash_log_tokens()
            self.metrics.crash_logs_processed += len(
                [r for r in suite.results if not r.passed]
            )
            self.metrics.crash_log_tokens += crash_tokens
            self.metrics.self_heal_iterations = iteration
            self.metrics.qa_summary = suite.summary()

            # Track context window for this QA cycle
            context_estimate = crash_tokens + estimate_tokens(code.source_code)
            self.metrics.largest_context_window = max(
                self.metrics.largest_context_window,
                context_estimate,
            )

            self.log.info("QA result: %s", suite.summary())
            self._write_checkpoint(f"04_qa_iter{iteration}.json", {
                "summary": suite.summary(),
                "crash_log_tokens": crash_tokens,
            })

            if suite.all_passed():
                self.log.info("All tests passed — self-healing complete.")
                break

            # ── Dev refactor ─────────────────────────────────────────
            all_logs = "\n".join(
                r.crash_log for r in suite.results if not r.passed
            )
            self.log.warning(
                "Feeding %d tokens of crash logs into Dev Agent for refactor",
                estimate_tokens(all_logs),
            )

            code = await self.dev.refactor(
                code=code,
                error_log=all_logs,
                review_feedback=code.review_comment,
            )

            self._write_checkpoint(f"05_refactored_iter{iteration}_{code.file_name}", code.source_code)
            self.log.info("Refactored — ~%d tokens", code.token_estimate())
        else:
            self.log.warning("Max iterations (%d) reached with failing tests", iteration)
            # Save failing test info for debugging
            self._write_checkpoint("_max_iterations_reached.txt",
                                   f"Self-heal stopped after {iteration} iterations.\n"
                                   f"Final QA summary: {suite.summary()}\n")

        # Final LLM metrics
        self.metrics.total_llm_calls = self.llm.call_count
        self.metrics.prompt_tokens = self.llm.total_prompt_tokens
        self.metrics.completion_tokens = self.llm.total_completion_tokens

        return code

    # ── Metrics & reporting ────────────────────────────────────────────────

    def _finalise_metrics(self) -> None:
        self.metrics.elapsed_seconds = time.time() - self._start_time

    def _log_report(self) -> None:
        snap = self.metrics.snapshot()
        self.log.info("")
        self.log.info("=" * 72)
        self.log.info("  PIPELINE COMPLETE — API-Token Consumption Report")
        self.log.info("  Run ID: %s", snap["run_id"])
        self.log.info("=" * 72)
        self.log.info("  Phases completed:        %s", ", ".join(snap["phases_completed"]))
        self.log.info("  Total LLM calls:         %d", snap["total_llm_calls"])
        self.log.info("  Prompt tokens:           %d", snap["prompt_tokens"])
        self.log.info("  Completion tokens:       %d", snap["completion_tokens"])
        self.log.info("  Total tokens consumed:   %d", snap["total_tokens"])
        self.log.info("  Largest context window:  %d tokens", snap["largest_context_window_tokens"])
        self.log.info("  Self-heal iterations:    %d", snap["self_heal_iterations"])
        self.log.info("  Crash logs processed:    %d", snap["crash_logs_processed"])
        self.log.info("  Crash log tokens:        %d tokens", snap["crash_log_tokens"])
        self.log.info("  Trends discovered:       %d", snap["scout_trends_found"])
        self.log.info("  Generated file:          %s", snap["generated_file"])
        self.log.info("  Elapsed time:            %.1f s", snap["elapsed_seconds"])
        self.log.info("=" * 72)

    # ── Checkpoints ────────────────────────────────────────────────────────

    def _write_checkpoint(self, name: str, content: str | dict) -> None:
        path = self._ckpt_dir / name
        try:
            if isinstance(content, dict):
                path.write_text(json.dumps(content, ensure_ascii=False, indent=2))
            else:
                path.write_text(content)
            self.log.debug("Checkpoint saved: %s", path)
        except OSError as exc:
            self.log.warning("Failed to write checkpoint %s: %s", path, exc)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = PipelineConfig()
    _setup_logging(cfg.log_level)

    asyncio.run(Orchestrator(cfg).run())


if __name__ == "__main__":
    main()
