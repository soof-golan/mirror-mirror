const canvas = document.getElementById('frame-canvas');
const ctx = canvas.getContext('2d');
const vignetteOverlay = document.getElementById('vignette-overlay');
const promptText = document.getElementById('prompt-text');

let currentImage = new Image();
let isReady = false;

function hslToRgba(h, s, l, a) {
    s /= 100;
    l /= 100;
    const k = n => (n + h / 30) % 12;
    const f = n => l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
    const r = Math.round(255 * f(0));
    const g = Math.round(255 * f(8));
    const b = Math.round(255 * f(4));
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function updateVignette(intensity, hue) {
    if (intensity < 0.001) {
        vignetteOverlay.style.boxShadow = 'none';
        return;
    }

    const opacity = Math.min(intensity * 2.0, 1.0);
    const spread = 100 + intensity * 200;
    const color = hslToRgba(hue, 100, 50, opacity);

    vignetteOverlay.style.boxShadow = `inset 0 0 ${spread}px ${spread / 2}px ${color}`;
}

function updateFrame(base64Data) {
    if (!base64Data) return;

    currentImage.onload = function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    };
    currentImage.src = 'data:image/jpeg;base64,' + base64Data;
}

function updatePrompt(text) {
    promptText.textContent = text || '';
}

async function pollUpdates() {
    if (!isReady) return;

    try {
        const state = await window.pywebview.api.get_all_state();

        if (state.frame) {
            updateFrame(state.frame);
        }
        if (state.vignette) {
            updateVignette(state.vignette.intensity, state.vignette.hue);
        }
        if (state.prompt !== undefined) {
            updatePrompt(state.prompt);
        }
    } catch (e) {
        console.error('Poll error:', e);
    }
}

function toggleFullscreen() {
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.toggle_fullscreen();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
    }
    if (e.key === 'Escape') {
        toggleFullscreen();
    }
});

document.addEventListener('dblclick', toggleFullscreen);

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

window.addEventListener('pywebviewready', function() {
    console.log('PyWebView ready');
    isReady = true;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    pollUpdates();

    setInterval(pollUpdates, 16);
});
