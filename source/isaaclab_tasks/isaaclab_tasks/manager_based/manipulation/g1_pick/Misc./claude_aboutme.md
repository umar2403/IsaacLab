# Claude Context File — Shahid Mulla

---

## CODING PREFERENCES

### Style
- Production-ready code — clean, minimal, no clutter
- Minimal but meaningful comments:
  - Docstrings on every non-trivial function (explain args, returns, and the math/idea behind it in 2-4 lines max)
  - Inline comments only when the line is non-obvious (e.g. a tricky indexing op or a math step)
  - NO section headers like "# ---- do thing ----" unless the file is long and genuinely needs structure
  - NO restating what the code obviously does (e.g. no "# increment counter" above n += 1)
- Self-explanatory variable names — prefer `world_T_imu` over `T` and `obs_idx` over `idx`

### File & Project Structure
- Each major component gets its own .py file (e.g. imu_ekf.py, landmark_ekf.py, vi_slam.py)
- Every file is independently runnable: has a `if __name__ == "__main__":` block that runs on all datasets and saves all outputs
- A `run(dataset_name, data_path, results_dir, ...)` function is the importable entry point — main.py just calls these
- Results are NEVER hardcoded into main.py — each file handles its own outputs
- All outputs go into a `results/` folder (created automatically if missing)
- Unique filenames: if a file already exists, append `_n` (smallest available integer) — never overwrite

### Results & Outputs
- Every run saves figures to disk — never just `plt.show()`
- Figures use a standard theme. No need to go out of the way to do something different or unnecessary.
- All plots are high quality: `dpi=150`, `bbox_inches="tight"`
- Print meaningful runtime info: shapes, final positions, key metrics — enough to debug without opening figures
- Save intermediate outputs as when downstream scripts need them

### Toggles & Tuning Parameters
- Put all toggleable booleans and tunable constants at the TOP of the file in a clearly marked block
- Example pattern:
  ```python
  USE_COMPUTED_FEATURES = False   # True → load from results/
  SIGMA_OBS = 10.0                # pixel std for observation noise
  MAX_LANDMARKS = 800             # subsample for tractability
  ```
- When two modes exist (e.g. computed vs provided features), output filenames should encode the toggle value
  e.g. `dataset00_landmarks_ekf_False.npy`

### Numerical / Math Code
- Always `np.seterr(divide='ignore', invalid='ignore')` at the top when dealing with SE(3) or projection math
- Prefer vectorised numpy ops; avoid Python loops over large arrays where possible
- Use named intermediate variables for Jacobian blocks — don't chain everything into one line

### Dependencies
- Standard stack: numpy, scipy, matplotlib, opencv-python, transforms3d etc.
- No unnecessary imports

---

## REPORT WRITING PREFERENCES

### Tone & Language
- Professional academic English — clear and precise, not verbose
- Simple sentence structures over complex nested clauses
- No filler phrases like "it is worth noting that" or "it can be seen that"
- Never throw in technical jargon just to sound impressive — every term used should be necessary

### Math
- Prefer equations over wordy explanations — if something can be said in math, say it in math
- Use proper LaTeX notation: `\boldsymbol{\Sigma}`, `\mathrm{tr}(\cdot)`, `\in \mathrm{SE}(3)`
- Inline math for quantities mentioned in text: $\sigma_{\text{obs}} = 70$~pixels
- Display equations for anything central to the method

### Structure
- Short, dense paragraphs — one idea per paragraph
- Results sections: state the number first, then interpret it, then explain why
- When comparing two methods/sources: parallel structure sentence by sentence
- Do not repeat the same point in different words — say it once, move on

### Figures
- Always referenced before they appear in text: Fig.~\ref{fig:xxx}
- Captions are self-contained — a reader should understand the figure without reading the body text
- For multi-panel figures: label each panel in the caption (Top: ... Bottom: ...)

### Citations
- Only cite things directly used or directly relevant — no padding
- Prefer: original algorithm papers, textbooks, official library papers, course notes
- Format: Author(s), Year, Title in italics, Venue/Publisher
- Lucas-Kanade covers optical flow — no need for a separate optical flow citation

### Conciseness Rule
- If asked to make something concise: cut words, not content
- Every sentence should earn its place — if removing it loses no information, remove it

---

## WORKFLOW PREFERENCES

### When given a new project
1. Read ALL provided files and specs before writing any code
2. Give a full plan (file structure + what each file does) before starting
3. Ask about toggles/dataset structure upfront if unclear
4. Implement one file at a time, in dependency order

### Error handling
- When there's a bug: read the traceback, identify the exact line and cause, give the minimal fix
- Don't rewrite the whole file for a one-line fix — just give the changed block with clear instructions on where to put it
- If a full rewrite IS needed, say why before doing it

### General
- Do not add unsolicited features or complexity — implement exactly what was asked
- If something is ambiguous, state the assumption made and proceed
- No excessive post-amble after delivering code or text — just deliver it