# Glaciology + Machine Learning

Summer-school lecture notebooks, rendered as a Quarto website.

Live site: <https://glamacles.github.io/jake_slides/>

- `UDE_glaciology_lecture.ipynb` — Universal Differential Equations for glaciology
- `AD_and_adjoints_concepts.ipynb` — differentiation, adjoints, and differentiable solvers
- `gaussian_processes.ipynb` — Gaussian processes

The `.ipynb` files are the source of truth. `_quarto.yml` defines the website; rendered output goes to `_site/` (gitignored).

## Local preview

```bash
source .venv/bin/activate
quarto preview          # live-reloading local server
# or
quarto render           # one-off build into _site/
```

## Deploy to GitHub Pages

Run from the **`main`** branch. You never check out `gh-pages` yourself — the publish step manages that branch. The site is served via Pages "deploy from a branch" (`gh-pages`, root).

Render locally (the JAX notebooks run on your machine, not in CI), then push the built site:

```bash
source .venv/bin/activate
quarto render --no-execute                       # rebuild _site from the notebooks' stored outputs
quarto publish gh-pages --no-render --no-prompt  # push _site to the gh-pages branch
```

Notes:

- `--no-execute` reuses the outputs already saved in each notebook (no retraining). Drop it and run a plain `quarto render` when you actually want the notebooks re-executed.
- Deploying only touches the `gh-pages` branch. Commit and push your source changes on `main` separately.
- Requires the git credential helper set up for pushes (`gh auth setup-git`).
