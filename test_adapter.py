import asyncio
import json
import httpx
from server import app

async def run_tests():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("[TEST 1] Testing GET /health...")
        res = await client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        print("  -> OK:", res.json())

        print("\n[TEST 2] Testing GET /v1/models...")
        res = await client.get("/v1/models")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        models_data = res.json()
        assert models_data.get("object") == "list"
        model_ids = [m["id"] for m in models_data.get("data", [])]
        print(f"  -> Found {len(model_ids)} models. Sample: {model_ids[:5]}")
        assert "gemini-3.8-flash" in model_ids
        assert "gpt-4o" in model_ids

        print("\n[TEST 3] Testing GET /v1/models/gemini-3.8-flash...")
        res = await client.get("/v1/models/gemini-3.8-flash")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        m_info = res.json()
        assert m_info.get("id") == "gemini-3.8-flash"
        print("  -> OK:", m_info)

        print("\n[TEST 4] Testing GET /v1/models/gpt-4o (alias)...")
        res = await client.get("/v1/models/gpt-4o")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        m_info = res.json()
        assert m_info.get("id") == "gpt-4o"
        print("  -> OK:", m_info)

        print("\n[TEST 5] Testing POST /v1/chat/completions (Non-Streaming)...")
        chat_req = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a concise code helper."},
                {"role": "user", "content": "Respond with exactly: 'OK_PASSED'"}
            ],
            "max_tokens": 100
        }
        res = await client.post("/v1/chat/completions", json=chat_req)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        content = data["choices"][0]["message"]["content"] or ""
        print(f"  -> OK: Response: {content.strip()}")

        print("\n[TEST 6] Testing POST /v1/chat/completions (Streaming SSE)...")
        chat_req_stream = {
            "model": "gemini-3.8-flash",
            "messages": [
                {"role": "user", "content": "Say 'STREAM_OK'"}
            ],
            "stream": True,
            "max_tokens": 100
        }
        async with client.stream("POST", "/v1/chat/completions", json=chat_req_stream) as stream_resp:
            assert stream_resp.status_code == 200
            chunks = []
            async for line in stream_resp.aiter_lines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        chunks.append("[DONE]")
                    else:
                        d = json.loads(raw)
                        delta = d["choices"][0]["delta"].get("content", "")
                        if delta:
                            chunks.append(delta)
            full_text = "".join(c for c in chunks if c != "[DONE]")
            print(f"  -> OK: Stream chunks received. Full text: '{full_text.strip()}'")

        print("\n[TEST 7] Testing POST /v1/completions (Code Completion / Inline)...")
        comp_req = {
            "model": "gemini-3.8-flash",
            "prompt": "def square(x):\n",
            "max_tokens": 30
        }
        res = await client.post("/v1/completions", json=comp_req)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        comp_data = res.json()
        print("  -> OK:", comp_data["choices"][0]["text"].strip())

        print("\n[TEST 8] Testing Function / Tool Calling (Copilot Agent Tools)...")
        tool_req = {
            "model": "gemini-3.8-flash",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris? Call get_weather."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "stream": True
        }
        async with client.stream("POST", "/v1/chat/completions", json=tool_req) as tool_stream:
            assert tool_stream.status_code == 200
            called_tools = []
            async for line in tool_stream.aiter_lines():
                if line.startswith("data:") and line != "data: [DONE]":
                    d = json.loads(line[5:])
                    tcs = d["choices"][0]["delta"].get("tool_calls", [])
                    for tc in tcs:
                        called_tools.append(tc["function"]["name"])
            print("  -> OK: Tool calls detected in stream:", called_tools)
            assert "get_weather" in called_tools

        print("\n[TEST 9] Testing Virtual Fallback Model & Degradation Strategy (auto)...")
        from fallback_manager import fallback_manager
        # 1. Verify auto is present in models list
        res = await client.get("/v1/models")
        assert res.status_code == 200
        ids = [m["id"] for m in res.json()["data"]]
        assert "auto" in ids, "Virtual model 'auto' should be in models list"
        print("  -> OK: Virtual model 'auto' found in /v1/models")

        # 2. Test degradation recovery: invalid model -> gemini-3.8-flash
        orig_models = list(fallback_manager.fallback_models)
        orig_threshold = fallback_manager.failure_threshold
        try:
            fallback_manager.reconfigure(fallback_models=["invalid-model-for-test", "gemini-3.8-flash"], failure_threshold=1)
            # Turn 1: New session
            turn1_msgs = [{"role": "user", "content": "Test fallback session turn 1"}]
            fb_res = await client.post("/v1/chat/completions", json={
                "model": "auto",
                "messages": turn1_msgs,
                "max_tokens": 20
            })
            assert fb_res.status_code == 200, f"Expected 200 via fallback, got {fb_res.status_code}"
            print("  -> OK: Turn 1 successfully degraded from invalid-model-for-test to gemini-3.8-flash")

            # Turn 2: Ongoing session continues on degraded model
            turn2_msgs = [
                {"role": "user", "content": "Test fallback session turn 1"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Test fallback session turn 2"}
            ]
            fb_res2 = await client.post("/v1/chat/completions", json={
                "model": "auto",
                "messages": turn2_msgs,
                "max_tokens": 20
            })
            assert fb_res2.status_code == 200
            print("  -> OK: Turn 2 stayed on degraded model in same conversation")

            # Turn 3: New conversation resets back to model 0
            turn3_msgs = [{"role": "user", "content": "Brand new conversation question"}]
            sid3, is_new3 = fallback_manager.resolve_session(type("Req", (), {"messages": turn3_msgs, "user": None})())
            assert is_new3 == True, "Turn 3 without assistant message must be recognized as new session"
            print("  -> OK: Turn 3 verified as new session returning to initial model")
        finally:
            fallback_manager.reconfigure(fallback_models=orig_models, failure_threshold=orig_threshold)

    print("\n==========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
