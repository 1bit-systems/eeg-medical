# OpenNeuro CLI (via bun) — setup
Run:  .bun/bin/bunx --bun @openneuro/cli <subcommand>
Login writes to ~/.openneuro.

`openneuro login` is interactive. To avoid pasting a secret into a tool call,
set the key once via env when you run it, or write it to ~/.openneuro by hand.
The CLI also honors the OPENNEURO_API_KEY env var for downloads.
