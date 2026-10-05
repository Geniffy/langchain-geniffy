# Changelog

## 0.1.0

The first release.

- `GeniffyMemory(space=...)`: middleware that puts what is known about the user in the system prompt before each
  model call, after your own, and saves each finished exchange. Geniffy is asked once per question however many
  steps a run takes; an agent goes on without memory when Geniffy can't be reached. Sync and async agents.
- `geniffy_tools(space=...)`: `recall` and `remember`, reading the user from the agent's runtime.
- `GeniffyRetriever(space=...)`: the memories that match a query, as Documents with when each was said and where
  it came from.
- A space is required, so one user's memory is never written to another's.
