(() => {
  'use strict';
  const demo = document.getElementById('demo-video');
  const pps = document.getElementById('pps-video');
  const chapterButtons = [...document.querySelectorAll('.chapters button')];
  const status = document.getElementById('video-status');
  function playFrom(time) {
    const seek = () => { demo.currentTime = time; demo.play().catch(() => { status.textContent = 'Press play on the video to start this chapter.'; }); };
    if (demo.readyState >= 1) seek();
    else { demo.addEventListener('loadedmetadata', seek, { once: true }); demo.load(); }
    pps.pause();
    document.getElementById('experiments').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    status.textContent = `Jumping to ${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}.`;
  }
  document.querySelectorAll('[data-seek]').forEach(el => el.addEventListener('click', e => { e.preventDefault(); playFrom(Number(el.dataset.seek)); }));
  demo.addEventListener('timeupdate', () => {
    let label = 'Full demonstration';
    chapterButtons.forEach(button => {
      const active = demo.currentTime >= Number(button.dataset.seek) && demo.currentTime < Number(button.dataset.end);
      button.classList.toggle('active', active);
      if (active) { button.setAttribute('aria-current', 'true'); label = button.childNodes[1].textContent.trim(); }
      else button.removeAttribute('aria-current');
    });
    document.getElementById('chapter-label').textContent = label;
  });
  demo.addEventListener('play', () => pps.pause());
  pps.addEventListener('play', () => demo.pause());
  [demo, pps].forEach(video => video.addEventListener('error', () => { status.textContent = 'The video could not load. Use its download link to watch it locally.'; }));
  const dialog = document.getElementById('figure-dialog');
  let returnFocus;
  document.querySelectorAll('[data-image]').forEach(button => button.addEventListener('click', () => {
    returnFocus = button;
    document.getElementById('figure-image').src = button.dataset.image;
    document.getElementById('figure-title').textContent = button.dataset.title;
    dialog.showModal();
  }));
  document.getElementById('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', e => { if (e.target === dialog) { const r = dialog.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close(); } });
  dialog.addEventListener('close', () => returnFocus?.focus());
  document.getElementById('copy-citation').addEventListener('click', async () => {
    const text = document.getElementById('bibtex').textContent;
    const button = document.getElementById('copy-citation');
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied ✓';
      document.getElementById('copy-status').textContent = 'BibTeX citation copied.';
      setTimeout(() => { button.textContent = 'Copy citation ⧉'; }, 2200);
    } catch {
      const range = document.createRange(); range.selectNodeContents(document.getElementById('bibtex'));
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = 'Select & copy';
      document.getElementById('copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    }
  });
  const navLinks = [...document.querySelectorAll('nav a')];
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) navLinks.forEach(link => link.classList.toggle('active', link.hash === '#' + entry.target.id));
  }), { rootMargin: '-10% 0px -60% 0px' });
  navLinks.forEach(link => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
})();
