#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# ---------------------------------------------------------------------------
# Student code: imports, constants, and helper functions
# ---------------------------------------------------------------------------
from langchain_deepseek import ChatDeepSeek
import httpx

# Override load_mcp_tools with a resilient wrapper: retry a few times,
# then fall back to an empty list (score_cvs uses its own direct HTTP client,
# so the LangChain-wrapped tools are not needed for actual verification).
_orig_load_mcp_tools = load_mcp_tools


async def load_mcp_tools() -> list[Any]:  # type: ignore[no-redef]
    for attempt in range(3):
        try:
            return await _orig_load_mcp_tools()
        except Exception:
            if attempt == 2:
                return []  # graceful fallback
            await asyncio.sleep(3)
    return []

_SYSTEM_PROMPT = """\
You are a KYC verification agent. You receive a candidate's CV text and their \
LinkedIn profile (JSON). Compare the CV claims against the LinkedIn profile.

CHECK THESE FIELDS FOR DISCREPANCIES:

1. City: CV city must match LinkedIn city.

2. Each job (match by company name):
   - Company: must match (fake employer = discrepancy).
   - Title: "Senior X" when LinkedIn title is "X" with seniority "senior" -> \
NOT a discrepancy. But "Senior X" when seniority is "junior" or "mid" -> \
IS a discrepancy (inflated title).
   - Start year: must match exactly. Shifted years = discrepancy.
   - End year: must match exactly.

3. Education:
   - Degree: wording OK. "Bachelor of Science"=BSc, "Master of Business \
Administration"=MBA, "Doctor of Philosophy"=PhD, "Master of Science"=MSc.
   - School: must match. Fake school = discrepancy.
   - Field: wording variations OK.
   - Graduation year: must match exactly.

4. Skills: each CV skill must correspond to a LinkedIn skill by meaning, \
not by exact string. Wording variations are NOT discrepancies: e.g. \
"UI/UX Design" = "UI/UX", "Microsoft PowerPoint" = "PowerPoint", \
"Data Analysis" = "Data Analytics". A CV skill with no corresponding \
LinkedIn skill at all (e.g. "Docker" when LinkedIn has no similar skill) = \
discrepancy. CV listing fewer skills than LinkedIn = NOT a discrepancy.

NEVER count as discrepancy: job descriptions, LinkedIn headline, hometown, \
fewer skills or fewer jobs listed on CV.

SCORING:
- ANY discrepancy found -> SCORE: 0.15
- No discrepancy, all verified -> SCORE: 0.85
- Cannot determine -> SCORE: 0.25

End your response with exactly: SCORE: <number>
Do not include years or IDs in your final response.
"""

_SCORE_RE = re.compile(r"SCORE:\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_ANY_NUM_RE = re.compile(r"([0-9]+\.?[0-9]*)")


def _extract_score(text: str) -> float:
    """Pull a reliability score from the model's response."""
    if not text:
        return 0.3
    m = _SCORE_RE.search(text)
    if m:
        val = float(m.group(1))
        return max(0.0, min(1.0, val))
    fallback = 0.3
    for m in _ANY_NUM_RE.finditer(text):
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            fallback = val
    return fallback


# ---------------------------------------------------------------------------
# Direct MCP HTTP client (more reliable than per-call LangChain sessions)
# ---------------------------------------------------------------------------

class _MCPClient:
    """Lightweight async client for the SocialGraph MCP server."""

    def __init__(self, url: str = MCP_URL):
        self._url = url
        self._headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._client: httpx.AsyncClient | None = None
        self._msg_id = 0

    async def __aenter__(self) -> "_MCPClient":
        self._client = httpx.AsyncClient(timeout=120.0)
        # Initialise MCP session with retry (shared server can be flaky)
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = await self._client.post(
                    self._url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "hw2", "version": "1.0"},
                        },
                    },
                    headers=self._headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"initialize returned {resp.status_code}")
                session_id = resp.headers.get("mcp-session-id", "")
                if session_id:
                    self._headers["Mcp-Session-Id"] = session_id
                # Send initialised notification
                await self._client.post(
                    self._url,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=self._headers,
                )
                return self
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    await asyncio.sleep(2 + attempt)  # 2s, 3s, 4s backoff
        # All retries exhausted — raise so caller can handle
        raise ConnectionError(f"MCP initialize failed after 4 attempts: {last_exc}")

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def call(self, tool_name: str, **arguments: Any) -> Any:
        """Call an MCP tool and return the parsed JSON result (with retry)."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self._msg_id += 1
                resp = await self._client.post(
                    self._url,
                    json={
                        "jsonrpc": "2.0",
                        "id": self._msg_id + 1,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    },
                    headers=self._headers,
                )
                data = resp.json()
                if "result" in data and "content" in data["result"]:
                    for block in data["result"]["content"]:
                        text = block.get("text") if isinstance(block, dict) else None
                        if text:
                            try:
                                return json.loads(text)
                            except (json.JSONDecodeError, TypeError):
                                return text
                if "error" in data:
                    return {"error": str(data["error"])}
                return data.get("result", data)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(2)
        return {"error": f"call failed after 3 attempts: {last_exc}"}

    async def search_linkedin(
        self, q: str, location: str | None = None, industry: str | None = None
    ) -> list[dict]:
        kwargs: dict[str, Any] = {"q": q}
        if location:
            kwargs["location"] = location
        if industry:
            kwargs["industry"] = industry
        result = await self.call("search_linkedin_people", **kwargs)
        if isinstance(result, list):
            return result
        return []

    async def get_linkedin_profile(self, person_id: int) -> dict | None:
        result = await self.call("get_linkedin_profile", person_id=person_id)
        if isinstance(result, dict) and "error" not in result:
            return result
        return None


def _clean_cv_text(text: str) -> str:
    """Remove markdown table formatting (pipes, separator rows) from CV text."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip table separator rows (| ---- | ---- |)
        if stripped.startswith("|") and "---" in stripped:
            continue
        # Convert table rows: | cell1 | cell2 | → cell1 cell2
        if stripped.startswith("|") and stripped.endswith("|"):
            inner = stripped[1:-1]
            cells = [c.strip() for c in inner.split("|")]
            cells = [c for c in cells if c]
            cleaned.append(" ".join(cells) if cells else "")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _extract_name_city_industry(text: str) -> tuple[str, str | None, str | None]:
    """Extract candidate name, city, and industry from CV text."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    name = lines[0] if lines else ""
    city = None
    industry = None
    for line in lines[:6]:
        if "|" in line:
            loc_part = line.split("|")[0].strip()
            city = loc_part.split(",")[0].strip()
            break
        if "," in line and len(line.split(",")) == 2:
            parts = line.split(",")
            if len(parts[0]) < 40 and not parts[0].isdigit():
                city = parts[0].strip()
                break
    # Industry: second line usually "X Professional" → industry = X
    if len(lines) >= 2:
        second = lines[1]
        if second.endswith("Professional"):
            industry = second[: -len("Professional")].strip()
    return name, city, industry


# Keep old name as alias for backward compatibility
def _extract_name_and_city(text: str) -> tuple[str, str | None]:
    n, c, _ = _extract_name_city_industry(text)
    return n, c


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    model = ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=4096,
    )
    return {"model": model, "tools": tools}


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    model = agent["model"]
    sem = asyncio.Semaphore(3)

    async def _verify_one(name: str, text: str) -> float:
        async with sem:
            # Clean table formatting and extract candidate info
            clean_text = _clean_cv_text(text)
            candidate_name, city, industry = _extract_name_city_industry(clean_text)

            # Retry the entire MCP+model pipeline up to 3 times
            for attempt in range(3):
                try:
                    async with _MCPClient() as mcp:
                        # --- Search LinkedIn: name + city + industry (narrowest) ---
                        results = await mcp.search_linkedin(
                            candidate_name, location=city, industry=industry
                        )

                        # Fallback: name + city (no industry)
                        if not results:
                            results = await mcp.search_linkedin(
                                candidate_name, location=city
                            )

                        # Fallback: name + industry (no location)
                        if not results:
                            results = await mcp.search_linkedin(
                                candidate_name, industry=industry
                            )

                        # Fallback: name only
                        if not results:
                            results = await mcp.search_linkedin(candidate_name)

                        if not results:
                            return 0.25  # cannot find candidate

                        # --- Pick best match: prefer industry + city match ---
                        person_id = None
                        # Try exact industry + city match
                        for r in results:
                            if not isinstance(r, dict):
                                continue
                            ind = str(r.get("industry", "")).lower()
                            loc = str(r.get("location", "")).lower()
                            if (
                                industry
                                and industry.lower() in ind
                                and city
                                and city.lower() in loc
                            ):
                                person_id = r.get("id")
                                break
                        # Try industry match only
                        if person_id is None:
                            for r in results:
                                if not isinstance(r, dict):
                                    continue
                                ind = str(r.get("industry", "")).lower()
                                if industry and industry.lower() in ind:
                                    person_id = r.get("id")
                                    break
                        # Try city match only
                        if person_id is None:
                            for r in results:
                                if not isinstance(r, dict):
                                    continue
                                loc = str(r.get("location", "")).lower()
                                if city and city.lower() in loc:
                                    person_id = r.get("id")
                                    break
                        # Fall back to first result
                        if person_id is None and results:
                            person_id = results[0].get("id") if isinstance(results[0], dict) else None

                        if person_id is None:
                            return 0.25

                        # --- Get full LinkedIn profile ---
                        profile = await mcp.get_linkedin_profile(person_id)
                        if not profile:
                            return 0.25

                    # --- Ask the model to compare CV vs LinkedIn ---
                    profile_json = json.dumps(profile, indent=2)
                    user_msg = (
                        f"CV to verify:\n{clean_text}\n\n"
                        f"LinkedIn Profile (JSON):\n{profile_json}\n\n"
                        f"Compare every claim in the CV against the LinkedIn "
                        f"profile. Check city, each job (company, title vs "
                        f"seniority, start/end years), education (degree, school, "
                        f"field, graduation year), and skills. Then output "
                        f"SCORE: <number>."
                    )
                    response = await model.ainvoke(
                        [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ]
                    )
                    content = getattr(response, "content", str(response))
                    return _extract_score(content)

                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(3 + attempt)  # 3s, 4s backoff
                        continue
                    return 0.3  # all retries exhausted

    names = list(cvs.keys())
    scores = await asyncio.gather(
        *[_verify_one(n, cvs[n]) for n in names]
    )
    return dict(zip(names, scores))


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
