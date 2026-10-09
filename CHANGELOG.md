# Changelog

## 0.2.0

- **The briefing.** Before each model call the model gets the user's briefing: where things stand, what is due or
  was promised, the rules that apply, what happened, then what is known that bears on their last message, each line
  dated. `project=` keeps it to one project, `budget_chars=` caps it, and `briefing=False` reads only what bears on
  the question, as 0.1 did. An account without the briefing switched on gets that too, and the briefing is tried
  again ten minutes later. A briefing with nothing in it yet says so, so the model never reads silence as
  permission to guess.
- **The whole run is saved, tool calls included**, into one memory for the whole conversation: what the user said,
  each tool the agent called and what it returned, and the answer. 0.1 saved only the last question and answer, each
  as a memory of its own. The conversation is the agent's LangGraph thread, or `session=` for an app that keeps the
  history itself; with neither, each run is a memory of its own.
- **Never sent twice, never lost.** Where saving got to is kept in the agent's state, so with a checkpointer each run
  sends only what is new, across restarts and on every server, and a save that fails goes with the next run. Over 500
  messages go 500 at a time. A tool's output and a call's input are cut to what Geniffy keeps before they are sent.
- Needs `geniffy` 0.4.0 or later.

## 0.1.0

The first release.

- `GeniffyMemory(space=...)`: middleware that puts what is known about the user in the system prompt before each
  model call, after your own, and saves each finished exchange. Geniffy is asked once per question however many
  steps a run takes; an agent goes on without memory when Geniffy can't be reached. Sync and async agents.
- `geniffy_tools(space=...)`: `recall` and `remember`, reading the user from the agent's runtime.
- `GeniffyRetriever(space=...)`: the memories that match a query, as Documents with when each was said and where
  it came from.
- A space is required, and a blank one (or None from a function of the runtime) is refused, so one user's
  memory is never written to another's or to your own.
