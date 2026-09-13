# 10. Web search as a separate service

Status: accepted

## Context

`agent_core/search/base.py` has held a `SearchProvider` contract with no implementation since the
tool surface was written. The only network reach the assistant had was Cursor's own built-in
search, which the agent invokes itself — and that turned out to be the expensive path. The YouTube
factcheck pass instructs the model to check up to eight claims; ACP executes one tool call at a
time per turn, so those eight became eight sequential searches with a reasoning step between each,
followed by a second turn to write the summary. The searches are independent of one another and
none of that serialisation buys anything.

Two other things pushed the provider out of the Core process. Provider credentials — a Brave
subscription key — would otherwise live next to the code the model's tool calls execute against.
And `web_fetch` is the one tool that can turn a URL found in an untrusted document into an outbound
request; the machine it is made from should be a decision, not a side effect of where the Core
happens to run.

## Decision

A fifth deploy unit, `web-search`, with one backend chosen by configuration — the hosted Brave
Search API or a self-hosted SearXNG instance — behind a token-authenticated HTTP API.

1. `POST /v1/search` takes a **batch**: `{"queries": [...], "limit": n, "lang": "ru"}`. The queries
   run side by side on a small thread pool and the answer keeps the order of the request. A query
   that fails upstream carries its own `error` while the rest still return, because a rate-limited
   fourth claim must not discard the three that came back.
2. Results are reduced to `title`, `url`, `excerpt` and nothing else, at both ends. Handing the
   model a provider payload means handing it ranking metadata and sponsored blocks in a shape
   where an instruction and a snippet look identical. The client drops unknown fields again, so a
   service that grows one does not silently start feeding it to the model.
3. Every response restates that the results are content and not instructions, next to the data
   rather than only in the prompt.
4. **No job registry.** Unlike `gpu-transcriber` and `handwriting-ocr`, every endpoint answers in
   the same request. Search is sub-second work and a poll interval would cost more than the search;
   submit-and-poll here would defeat the reason the unit exists.
5. A TTL cache (default 15 minutes, LRU-bounded) sits in front of the provider. Repeats are common
   — a second turn re-checks a disputed number, a retried job repeats the batch, two videos cite
   the same study — and none of that should become another billed round trip. Whitespace and case
   do not change what an engine returns, so they do not miss. A different `limit` is a different
   entry.
6. `POST /v1/fetch` returns readable text for one page, HTML reduced to prose on the standard
   library and truncated to `SEARCH_FETCH_MAX_CHARS` (20 000). It can be switched off entirely,
   leaving the assistant with snippets and no way to open a page.
   Extraction is our own parser rather than a hosted reader. SearXNG has no endpoint for it —
   `/search`, `/autocompleter`, `/config`, `/image_proxy`, `/stats` and static assets are the whole
   surface — and adding an outside reader would put every page the assistant opens through a third
   party we do not otherwise talk to. Instead the parser prefers whatever the page marks as its
   main content, drops chrome by tag and by class name, and falls back a step at a time: a
   heuristic that empties a page yields the page with its menus rather than nothing.
7. The fetch guard is duplicated on purpose: `agent_core.search.base.guard_url` before the Core
   calls, `web_search.urls.guard_url` before the service connects. The two run on different hosts,
   so the loopback interface each protects is a different one — the Core's includes its own MCP
   endpoint, the service's may include the SearXNG instance.

   The service's guard runs again on **every redirect target**, not just the URL it was handed.
   `urllib` follows redirects without asking, so a page on a public host could otherwise answer
   `302` to `127.0.0.1` and walk straight through the check. It also returns the URL in ASCII form,
   because that is the only form `urllib` will send and the links worth opening in a Russian
   conversation are mostly Cyrillic Wikipedia titles.
8. `SEARCH_ENABLED=false` is the default and leaves `web_search` and `web_fetch` unregistered, so
   an unconfigured Core has no network reach through MCP at all.

   The YouTube factcheck reaches them through a **scoped MCP token**: `issue_scoped_token` binds a
   context and an allowlist to the token itself, and the server filters both `tools/list` and
   `tools/call` by it. That pass used to be created with `mcp_servers=[]`, which left it with no
   search at all and only the agent's built-in one. Handing it the ordinary session entry instead
   would have been worse: it reads a transcript nobody vouched for, and that transcript should not
   be one prompt away from the calendar. The token is released as soon as the pass ends, so the
   second pass — which must not search — cannot use it either.
9. The unit has **no runtime dependencies**. Its own HTTP surface and its outbound calls are both
   standard library, so it installs beside the Core without a venv or a build toolchain.

`RemoteSearchProvider.search` satisfies the existing protocol for the MCP tool, where the agent
asks one question at a time because that is what a tool call is. `search_many` is for the Core's
own passes, which know the whole list up front.

## Consequences

- Provider credentials live in one process that knows nothing about the assistant, on a service
  that binds loopback by default.
- A Core-side factcheck can collapse eight round trips into one. Restructuring the YouTube prompt
  to extract claims first and search them as a batch is now possible; it is not done here.
- Brave `/health` cannot be honest for free: a probe would spend quota, so it reports the outcome
  of the last real query. SearXNG probes `/config`.
- Switching backends is a configuration change, and the tests never reach the network — a fake
  backend sits behind the real HTTP server.
- Search results remain `UNTRUSTED_CONTENT`. Nothing inferred from them is written without
  confirmation (ADR 0007).
