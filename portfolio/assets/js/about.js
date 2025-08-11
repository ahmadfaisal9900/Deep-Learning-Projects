async function renderAbout() {
  const root = document.getElementById('about');
  try {
    const res = await fetch('data/profile.json');
    const profile = await res.json();

    const linksHtml = (profile.links || [])
      .map((l) => `<a class="chip" target="_blank" rel="noopener" href="${l.url}">${l.label}</a>`) 
      .join('');

    const skillsHtml = (profile.skills || [])
      .map((s) => `<span class="chip">${s}</span>`) 
      .join('');

    root.innerHTML = `
      <div class="card">
        <h2 style="margin:0 0 6px;">${profile.name}</h2>
        <p class="pub-meta" style="margin:0 0 10px;">${profile.role}${profile.affiliation ? ' · ' + profile.affiliation : ''}${profile.location ? ' · ' + profile.location : ''}</p>
        <p>${profile.summary || ''}</p>
        <div style="margin:10px 0 0; display:flex; gap:8px; flex-wrap:wrap;">${linksHtml}</div>
      </div>
      <div class="card">
        <h3>Skills & Tools</h3>
        <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">${skillsHtml}</div>
      </div>
    `;
  } catch (err) {
    console.error(err);
    root.innerHTML = '<p>Could not load profile.</p>';
  }
}

renderAbout();