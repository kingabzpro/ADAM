import json

import httpx

from adam.decisions import Agent, Decision

DEFAULT_MODEL = "d3:2b"


class D3Client:
    def __init__(self, http: httpx.AsyncClient, endpoint: str = "http://127.0.0.1:11434", model: str = DEFAULT_MODEL):
        url = httpx.URL(endpoint)
        if url.scheme != "http" or url.host not in {"127.0.0.1", "::1"} or url.userinfo or url.query or url.fragment or url.path not in ("", "/"):
            raise ValueError("D3 endpoint must be a loopback HTTP origin")
        self.http = http
        self.endpoint = str(url).rstrip("/")
        self.model = model

    async def assess(self, ticker: str, agent: Agent, evidence: dict) -> Decision:
        if not ticker.strip() or not evidence:
            raise ValueError("Ticker and evidence are required")
        state = json.dumps({"ticker": ticker, "category": agent, "evidence": evidence}, allow_nan=False)
        response = await self.http.post(f"{self.endpoint}/v1/systemone", json={
            "model": self.model,
            "state": state,
            "questions": {"signal": {
                "type": "choice",
                "instructions": "Based only on this evidence, which research signal is best supported?",
                "criteria": {"BUY": "Evidence favors buying", "HOLD": "Evidence is mixed or neutral", "SELL": "Evidence favors selling"},
            }},
        }, follow_redirects=False)
        if response.status_code == 404:
            raise RuntimeError('Ollama model or decision endpoint unavailable; run uv run adam-setup and restart Ollama if needed')
        response.raise_for_status()
        try:
            answer = response.json()["answers"]["signal"]
            result = Decision(ticker=ticker, agent=agent, signal=answer["choice"],
                              probabilities=answer["probabilities"],
                              summary="D3 classification based on supplied evidence; scores are not profit probabilities.",
                              status="completed")
            if result.probabilities[result.signal] < max(result.probabilities.values()):
                raise ValueError("D3 choice does not match its scores")
            return result
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid D3 response") from error
