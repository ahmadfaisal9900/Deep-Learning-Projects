async function renderPublications() {
  const root = document.getElementById('publications');
  try {
    const res = await fetch('data/publications.json');
    const pubs = await res.json();
    root.innerHTML = pubs
      .sort((a, b) => (b.year || 0) - (a.year || 0))
      .map((p) => {
        const authors = Array.isArray(p.authors) ? p.authors.join(', ') : (p.authors || '');
        const links = (p.links || [])
          .map((l) => `<a class="chip" target="_blank" rel="noopener" href="${l.url}">${l.label}</a>`) 
          .join('');
        return `
          <div class="pub-item">
            <div class="pub-title">${p.title}</div>
            <div class="pub-meta">${authors}${p.venue ? ' · ' + p.venue : ''}${p.year ? ' · ' + p.year : ''}</div>
            <div class="pub-links">${links}</div>
          </div>
        `;
      })
      .join('');
  } catch (err) {
    console.error(err);
    root.innerHTML = '<p>Could not load publications.</p>';
  }
}

renderPublications();