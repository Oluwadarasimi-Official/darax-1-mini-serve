# DARAX-1 Mini inference service

Serves David's from-scratch 17.4M-parameter language model over HTTP.
Native weights only — zero external AI calls.

- `GET /health` — model status
- `POST /generate` — `{prompt, max_new_tokens, temperature, top_k}`

Deployed on Render (free tier). Used by the Official DARAX API as `darax-1-mini`.
