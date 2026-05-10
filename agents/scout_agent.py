"""
Scout Agent — Web intelligence gatherer for game trends.

Uses **Playwright** (headless Chromium) to scrape short-video platforms like
Douyin / TikTok and extract trending game mechanics.  When Playwright is
unavailable or the network is unreachable, falls back to a bundled data cache
so the pipeline never hard-fails.

The real Playwright flow (skipped in fallback mode):
  1. Launch headless Chromium.
  2. Navigate to ``https://www.douyin.com/hashtag/{hashtag}``.
  3. Scroll, wait for video-card elements, extract metadata.
  4. Return structured ``GameTrend`` objects.

Token note:
  Each scraped trend adds ~100–300 tokens to the context window.  A full
  scout run (3 hashtags, ~5 trends each) produces 1k–4k tokens of structured
  data that flows verbatim into the Dev Agent's prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.llm import LLMClient, estimate_tokens

logger = logging.getLogger("agents.scout")

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

_GAME_MECHANICS = [
    "match-3", "merge", "2048-like", "消除", "puzzle",
    "hyper-casual", "board-game", "candy-crush-like", "simulation", "idle",
]

_TREND_CACHE_PATH = Path(__file__).parent / "_trend_cache.json"


@dataclass
class GameTrend:
    """A single observed trend scraped from a short-video platform."""

    platform: str
    hashtag: str
    title: str
    engagement_score: float          # 0 – 100
    mechanic_tags: list[str]
    raw_snippet: str                  # Raw video-description text

    def token_estimate(self) -> int:
        return estimate_tokens(json.dumps(self.__dict__, ensure_ascii=False))

    def short_summary(self) -> str:
        return f"[{self.hashtag}] {self.title} — tags: {', '.join(self.mechanic_tags)}"


@dataclass
class ScoutReport:
    """Aggregated output from a full reconnaissance run."""

    trends: list[GameTrend] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(t.token_estimate() for t in self.trends)

    def to_llm_context_block(self) -> str:
        """XML-tagged block suitable for injecting into a system prompt."""
        lines = ["<scout_report>"]
        for t in self.trends:
            lines.append(
                f'  <trend platform="{t.platform}" hashtag="{t.hashtag}" '
                f'score="{t.engagement_score:.1f}">'
            )
            lines.append(f"    <title>{t.title}</title>")
            lines.append(f"    <mechanics>{', '.join(t.mechanic_tags)}</mechanics>")
            lines.append(f"    <snippet>{t.raw_snippet[:300]}</snippet>")
            lines.append("  </trend>")
        lines.append("</scout_report>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class ScoutAgent:
    """Gathers game-trend intelligence from short-video platforms.

    Parameters
    ----------
    llm:
        Shared LLM client (optional; used for post-scrape summarisation).
    data_dir:
        Where to cache scraped results for offline fallback.
    use_playwright:
        Set ``False`` to skip real browser scraping and use cached data.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        data_dir: str = "data/scout",
        use_playwright: bool = True,
    ) -> None:
        self.llm = llm
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_playwright = use_playwright and not os.getenv("SCOUT_NO_PLAYWRIGHT")
        logger.info(
            "ScoutAgent initialised [use_playwright=%s, data_dir=%s]",
            self.use_playwright,
            self.data_dir,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def recon(
        self,
        platform: str = "douyin",
        hashtags: Optional[list[str]] = None,
    ) -> ScoutReport:
        """Run a full reconnaissance cycle.

        Tries real Playwright scraping first; falls back to cached data on
        any failure.  Uses the LLM (if available) to enrich raw snippets into
        structured trends.
        """
        hashtags = hashtags or ["超解压三消", "益智游戏开发", "cocoscreator"]
        logger.info("ScoutAgent.recon — platform=%s, hashtags=%s", platform, hashtags)

        if self.use_playwright:
            try:
                trends = await self._scrape_with_playwright(platform, hashtags)
                if trends:
                    self._cache_trends(trends)
                    logger.info("Scraped %d trends via Playwright", len(trends))
                    return ScoutReport(trends=trends)
            except Exception as exc:
                logger.warning("Playwright scrape failed: %s — falling back to cache", exc)

        # Playwright unavailable or failed → load cached data
        cached = self._load_cached_trends()
        if cached:
            logger.info("Loaded %d trends from cache", len(cached))
            return ScoutReport(trends=cached)

        # No cache either → build a deterministic default report
        logger.warning("No cached data — building default report")
        return self._default_report(platform)

    # ── Playwright scraping ─────────────────────────────────────────────────

    async def _scrape_with_playwright(
        self,
        platform: str,
        hashtags: list[str],
    ) -> list[GameTrend]:
        """Concurrent Playwright scrape across multiple hashtags."""
        try:
            from playwright.async_api import async_playwright  # slow import
        except ImportError:
            logger.warning("playwright not installed — skipping real scrape")
            raise ImportError("playwright not installed")

        trends: list[GameTrend] = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )

            tasks = [self._scrape_hashtag(context, platform, tag) for tag in hashtags]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Scrape task failed: %s", r)
                elif r:
                    trends.extend(r)

            await browser.close()

        # If we have an LLM, enrich the raw snippets into structured trends
        if self.llm and trends:
            trends = await self._enrich_with_llm(trends, platform)

        return trends

    async def _scrape_hashtag(
        self,
        context: "playwright.async_api.BrowserContext",  # type: ignore[name-defined]
        platform: str,
        hashtag: str,
    ) -> list[GameTrend]:
        """Scrape a single hashtag page and extract video metadata."""
        page = await context.new_page()
        try:
            url = f"https://www.douyin.com/hashtag/{hashtag}"
            logger.info("Navigating to %s", url)

            await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
            await page.wait_for_timeout(3000)  # let JS render

            # Scroll to trigger lazy-load
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, 800)")
                await page.wait_for_timeout(1000)

            # Extract video cards (Douyin structure approximates this)
            cards = await page.query_selector_all(".video-card, .xgplayer")
            if not cards:
                # Fallback: extract from page title / meta
                title = await page.title()
                desc = await page.evaluate(
                    "() => document.querySelector('meta[name=description]')?.content || ''"
                )
                if desc:
                    trends = self._parse_meta_trends(platform, hashtag, title, desc)
                else:
                    logger.warning("No content extracted for %s", hashtag)
                    trends = []
            else:
                trends = []
                limit = min(len(cards), 8)
                for card in cards[:limit]:
                    trend = await self._extract_card(card, platform, hashtag)
                    if trend:
                        trends.append(trend)

            logger.info("Scraped %d items from %s", len(trends), hashtag)
            return trends

        finally:
            await page.close()

    async def _extract_card(
        self,
        card: "playwright.async_api.ElementHandle",  # type: ignore[name-defined]
        platform: str,
        hashtag: str,
    ) -> Optional[GameTrend]:
        """Extract trend data from a single video-card element."""
        try:
            title_el = await card.query_selector(".title, .video-title, h3")
            title = await title_el.inner_text() if title_el else hashtag
            title = title.strip()[:120]

            desc_el = await card.query_selector(".desc, .video-desc, p")
            snippet = await desc_el.inner_text() if desc_el else ""
            snippet = snippet.strip()[:400]

            # Simulate an engagement score from available signals
            likes_el = await card.query_selector(".like-count, .engage-count")
            likes_text = await likes_el.inner_text() if likes_el else "0"
            likes = self._parse_count(likes_text)
            engagement_score = min(100.0, likes / 5000)

            return GameTrend(
                platform=platform,
                hashtag=hashtag,
                title=title or hashtag,
                engagement_score=round(engagement_score, 1),
                mechanic_tags=self._infer_mechanics(title + " " + snippet),
                raw_snippet=snippet,
            )
        except Exception as exc:
            logger.debug("Failed to extract card: %s", exc)
            return None

    # ── LLM enrichment ─────────────────────────────────────────────────────

    async def _enrich_with_llm(
        self,
        trends: list[GameTrend],
        platform: str,
    ) -> list[GameTrend]:
        """Let the LLM classify mechanics and assign engagement scores."""
        snippets = "\n".join(
            f"[{i}] {t.title} | {t.raw_snippet[:150]}" for i, t in enumerate(trends)
        )

        prompt = textwrap.dedent(f"""\
            You are a game-trend analyst.  Classify each of the following
            snippets from {platform} into game mechanics.

            For each item, output a JSON line:
              {{"index": <i>, "mechanics": ["match-3", …], "score": <0-100>}}

            Snippets:
            {snippets}
        """)

        reply = await self.llm.chat_simple(prompt) if self.llm else None
        if not reply:
            return trends

        try:
            for line in reply.strip().splitlines():
                data = json.loads(line)
                idx = data.get("index")
                if idx is not None and idx < len(trends):
                    trends[idx].mechanic_tags = data.get("mechanics", trends[idx].mechanic_tags)
                    trends[idx].engagement_score = float(data.get("score", trends[idx].engagement_score))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("LLM enrichment parse failed: %s", exc)

        return trends

    # ── Fallback data ───────────────────────────────────────────────────────

    def _default_report(self, platform: str) -> ScoutReport:
        """Deterministic default when both Playwright and cache are unavailable."""
        trends = [
            GameTrend(
                platform=platform,
                hashtag="#超解压三消",
                title="Match-3 Puzzle with Chain Explosion Mechanics",
                engagement_score=92.3,
                mechanic_tags=["match-3", "puzzle", "candy-crush-like"],
                raw_snippet=(
                    "超解压三消游戏，连锁爆炸消除，特殊方块合成。"
                    "Engagement: 234k likes, 3400 comments."
                ),
            ),
            GameTrend(
                platform=platform,
                hashtag="#益智游戏开发",
                title="Merge & Idle Hybrid Gameplay",
                engagement_score=78.5,
                mechanic_tags=["merge", "idle", "simulation"],
                raw_snippet=(
                    "益智游戏开发教程：从零开始制作合并类游戏，"
                    "使用 Cocos Creator 实现拖拽合并逻辑。"
                ),
            ),
            GameTrend(
                platform=platform,
                hashtag="#cocoscreator",
                title="Cocos Creator 3.x Match-3 Board Tutorial",
                engagement_score=85.1,
                mechanic_tags=["match-3", "board-game", "cocoscreator"],
                raw_snippet=(
                    "Cocos Creator 3.x match-3 game tutorial series. "
                    "Covers grid generation, swap mechanics, and touch input handling."
                ),
            ),
        ]
        return ScoutReport(trends=trends)

    def _cache_trends(self, trends: list[GameTrend]) -> None:
        """Persist scraped trends for offline use."""
        data = [t.__dict__ for t in trends]
        _TREND_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Cached %d trends to %s", len(trends), _TREND_CACHE_PATH)

    def _load_cached_trends(self) -> list[GameTrend]:
        """Load previously cached trends."""
        if not _TREND_CACHE_PATH.exists():
            return []
        try:
            data = json.loads(_TREND_CACHE_PATH.read_text())
            return [GameTrend(**item) for item in data]
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Cache read failed: %s", exc)
            return []

    # ── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_meta_trends(
        platform: str,
        hashtag: str,
        title: str,
        desc: str,
    ) -> list[GameTrend]:
        """Fallback: extract trend from page <title> / <meta> when no cards found."""
        return [
            GameTrend(
                platform=platform,
                hashtag=hashtag,
                title=title[:120],
                engagement_score=50.0,
                mechanic_tags=ScoutAgent._infer_mechanics(title + " " + desc),
                raw_snippet=desc[:400],
            )
        ]

    @staticmethod
    def _infer_mechanics(text: str) -> list[str]:
        """Simple keyword-based mechanic inference."""
        text_lower = text.lower()
        found = []
        for mech in _GAME_MECHANICS:
            if mech.lower() in text_lower:
                found.append(mech)
        return found or ["puzzle"]

    @staticmethod
    def _parse_count(text: str) -> float:
        """Parse human-readable counts like '12.3k', '4.5M' into float."""
        text = text.strip().replace(",", "")
        if text.endswith("M"):
            return float(text[:-1]) * 1_000_000
        if text.endswith("k"):
            return float(text[:-1]) * 1_000
        if text.endswith("万"):
            return float(text[:-1]) * 10_000
        try:
            return float(text)
        except ValueError:
            return 0.0
