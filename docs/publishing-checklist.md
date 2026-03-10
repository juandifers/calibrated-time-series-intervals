# Publishing Checklist

## Before Making The Repository Public

- remove `Final Internship Report.pdf` from the published repo
- choose and add a repository license
- decide where the large model artifact should live: GitHub release asset is the simplest default for this repo; add a checksum manifest and download helper if you move it out of git
- run `python scripts/render_public_notebooks.py` locally and keep the generated HTML if you want no-Jupyter browsing
- capture 2-3 screenshots for the README: one EDA plot, one replay/backtest view, one dashboard screenshot
- test the quickstart commands from a clean virtual environment

## Recommended Final Polish

- add a short description and topics on the GitHub repository page
- pin the most important screenshot near the top of the README
- include this project near the top of your CV with emphasis on calibrated intervals, heteroscedasticity, and reliability-oriented forecasting
- if you publish a live demo, make it explicit that it is a replay service on anonymized data
