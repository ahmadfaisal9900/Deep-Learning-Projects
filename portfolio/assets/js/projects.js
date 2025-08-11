let allProjects = [];
let activeTag = 'All';

function renderFilters(projects) {
  const filters = document.getElementById('project-filters');
  if (!filters) return;
  const tagSet = new Set(['All']);
  projects.forEach((p) => (p.tags || []).forEach((t) => tagSet.add(t)));
  filters.innerHTML = '';
  [...tagSet].forEach((tag) => {
    const btn = document.createElement('button');
    btn.className = 'filter-btn' + (tag === activeTag ? ' active' : '');
    btn.textContent = tag;
    btn.addEventListener('click', () => {
      activeTag = tag;
      renderFilters(allProjects);
      renderProjects(allProjects);
    });
    filters.appendChild(btn);
  });
}

function renderProjects(projects) {
  const root = document.getElementById('projects');
  if (!root) return;
  const filtered = activeTag === 'All' ? projects : projects.filter((p) => (p.tags || []).includes(activeTag));
  root.innerHTML = filtered
    .map((p) => {
      const tags = (p.tags || []).map((t) => `<span class="chip">${t}</span>`).join('');
      const links = (p.links || [])
        .map((l) => `<a class="chip" target="_blank" rel="noopener" href="${l.url}">${l.label}</a>`) 
        .join('');
      return `
        <article class="card">
          <h3>${p.title}</h3>
          <p>${p.description || ''}</p>
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin:8px 0;">${tags}</div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">${links}</div>
        </article>
      `;
    })
    .join('');
}

async function initProjects() {
  try {
    const res = await fetch('data/projects.json');
    const projects = await res.json();
    allProjects = projects;
    renderFilters(projects);
    renderProjects(projects);
  } catch (err) {
    console.error(err);
    const root = document.getElementById('projects');
    if (root) root.innerHTML = '<p>Could not load projects.</p>';
  }
}

initProjects();