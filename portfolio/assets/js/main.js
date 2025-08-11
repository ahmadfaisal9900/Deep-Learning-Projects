(function () {
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  const savedTheme = localStorage.getItem('theme');
  const isLight = savedTheme ? savedTheme === 'light' : prefersLight;
  if (isLight) document.documentElement.classList.add('light');

  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      document.documentElement.classList.toggle('light');
      const nowLight = document.documentElement.classList.contains('light');
      localStorage.setItem('theme', nowLight ? 'light' : 'dark');
      toggle.textContent = nowLight ? '☀' : '☾';
    });
    toggle.textContent = document.documentElement.classList.contains('light') ? '☀' : '☾';
  }

  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  const path = location.pathname;
  document.querySelectorAll('.site-nav .nav-link').forEach((link) => {
    const href = (link.getAttribute('href') || '').replace(location.origin, '');
    if (href && path.endsWith(href.split('/').pop() || 'index.html')) {
      link.classList.add('active');
    }
  });
})();