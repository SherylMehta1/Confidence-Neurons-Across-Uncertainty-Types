# Environment Setup

## 1. HuggingFace access
1. Request access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace (gated model, approval can take a few days — do this first).
2. Generate a HuggingFace access token (Settings -> Access Tokens -> New token, "read" permission).
3. **Do not commit your token to git.** On Kaggle, use Kaggle Secrets:
   ```python
   from kaggle_secrets import UserSecretsClient
   hf_token = UserSecretsClient().get_secret("HF_TOKEN")
   from huggingface_hub import login
   login(token=hf_token)
   ```
   Locally, use a `.env` file (already in `.gitignore`) or an environment variable.

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Kaggle-specific notes
- Turn ON "Internet" in notebook settings (needed for pip installs + model download).
- Verify your phone number on Kaggle to unlock GPU quota (30 hrs/week).
- Start with 4-bit quantization for early pipeline testing (fits in 16GB). Move precision-sensitive
  Phase 3/4 runs to a paid GPU rental (RunPod/Vast.ai) if full precision is needed.
- Sessions disconnect after ~12 hours — save intermediate results to files regularly, don't rely on
  keeping everything in notebook memory.

## 4. First thing everyone should run
`notebooks/00_hello_world.ipynb` — loads the model, runs a forward pass, computes entropy on two
test prompts, and confirms the ambiguous prompt has higher entropy than the confident one. If this
runs cleanly, your environment is ready.
