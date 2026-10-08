(() => {
  'use strict';
  const videos = [...document.querySelectorAll('video')];
  const dialog = document.getElementById('media-dialog');
  const expanded = document.getElementById('expanded-video');
  const expandedImage = document.getElementById('expanded-image');
  const status = document.getElementById('video-status');
  let returnFocus;
  let inlineSource;

  function play(video) {
    video.controls = true;
    video.closest('.clip-screen')?.classList.add('started');
    video.play().catch(() => {
      status.textContent = 'Use the video controls to start playback.';
    });
  }

  document.querySelectorAll('[data-play]').forEach(button => {
    button.addEventListener('click', () => play(document.getElementById(button.dataset.play)));
  });

  videos.forEach(video => {
    if (video.closest('.clip-screen')) video.controls = false;
    video.addEventListener('play', () => {
      videos.forEach(other => { if (other !== video) other.pause(); });
      video.closest('.clip-screen')?.classList.add('started');
    });
    video.addEventListener('error', () => {
      status.textContent = 'This video could not load. You can download its MP4 using the link below.';
    });
  });

  document.querySelectorAll('[data-expand]').forEach(button => button.addEventListener('click', () => {
    returnFocus = button;
    inlineSource = document.getElementById(button.dataset.expand);
    const time = inlineSource.currentTime;
    inlineSource.pause();
    expandedImage.hidden = true;
    expanded.hidden = false;
    document.getElementById('dialog-help').hidden = false;
    document.getElementById('dialog-title').textContent = inlineSource.getAttribute('aria-label') + ' · PRISM Inference · 1×';
    expanded.poster = inlineSource.poster;
    expanded.src = inlineSource.querySelector('source').getAttribute('src');
    expanded.addEventListener('loadedmetadata', () => { expanded.currentTime = time; }, { once: true });
    dialog.showModal();
    play(expanded);
  }));

  document.querySelectorAll('[data-image]').forEach(button => button.addEventListener('click', () => {
    returnFocus = button;
    inlineSource = null;
    videos.forEach(video => video.pause());
    expanded.hidden = true;
    expandedImage.hidden = false;
    expandedImage.src = button.dataset.image;
    document.getElementById('dialog-help').hidden = true;
    document.getElementById('dialog-title').textContent = button.dataset.title;
    dialog.showModal();
  }));

  document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    expanded.pause();
    if (inlineSource && inlineSource.readyState >= 1 && Number.isFinite(expanded.currentTime)) {
      inlineSource.currentTime = Math.min(expanded.currentTime, inlineSource.duration);
    }
    expanded.removeAttribute('src');
    expanded.load();
    inlineSource = null;
    returnFocus?.focus();
  });

  document.getElementById('copy-citation').addEventListener('click', async () => {
    const button = document.getElementById('copy-citation');
    const citation = document.getElementById('bibtex');
    try {
      await navigator.clipboard.writeText(citation.textContent);
      button.textContent = 'Copied ✓';
      document.getElementById('copy-status').textContent = 'BibTeX citation copied.';
      setTimeout(() => { button.textContent = 'Copy citation ⧉'; }, 2200);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Select & copy';
      document.getElementById('copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    }
  });

  const links = [...document.querySelectorAll('nav a, .experiment-index a')];
  const sections = [...new Set(links.map(link => document.querySelector(link.hash)).filter(Boolean))];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      links.forEach(link => {
        const active = link.hash === '#' + entry.target.id;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-15% 0px -60% 0px' });
  sections.forEach(section => observer.observe(section));
})();
