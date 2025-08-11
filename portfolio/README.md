# ML/Data Science Portfolio

A minimalist, responsive portfolio for machine learning and data science. Multipage static site with About, Projects, and Publications.

## Quick start

Option 1: Open `index.html` directly in your browser.

Option 2: Serve locally (recommended for fetching JSON):

```bash
cd /workspace/portfolio
python3 -m http.server 8000
# then open http://localhost:8000/portfolio/index.html
```

If your server root is `/workspace`, visit `http://localhost:8000/portfolio/`.

## Customize

- Profile: edit `data/profile.json`
- Projects: edit `data/projects.json`
- Publications: edit `data/publications.json`
- Styling: edit `assets/css/styles.css`
- Header/footer or copy tweaks: edit `*.html` and `assets/js/*.js`

## Structure

- `index.html` — home
- `about.html` — About Me rendered from `data/profile.json`
- `projects.html` — Projects rendered from `data/projects.json` with tag filters
- `publications.html` — Publications rendered from `data/publications.json`
- `assets/css/styles.css` — theme and layout
- `assets/js/main.js` — theme toggle, active nav, year
- `assets/js/about.js` — render About
- `assets/js/projects.js` — render Projects + filters
- `assets/js/publications.js` — render Publications
- `assets/img/*` — icons/images

## Notes

- The site supports light/dark theme; it remembers your preference.
- JSON fetches require using a local server (file:// may block fetch in some browsers).
- Replace placeholder links with your actual GitHub, Scholar, and paper URLs.