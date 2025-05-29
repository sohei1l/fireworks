# Magic Brush

A browser playground where you fling glowing particles that bounce, fade, and chain-react. Built with Rust + WebGL to explore WASM in the browser.

## 🎮 Live Demo

[![Demo](https://img.shields.io/badge/Live%20Demo-Magic%20Brush-blue?style=for-the-badge)](https://sohei1l.github.io/magic-brush/)

## Quick Start

### Prerequisites

```bash
# Install Rust (if you don't have it)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
# OR: cargo install wasm-pack
```

### Build and Run

```bash
# Clone the repository
git clone https://github.com/sohei1l/neon-particles.git
cd neon-particles

# Build the WASM module
./build.sh
# OR: wasm-pack build --target web --out-dir pkg

# Serve locally (choose one)
python -m http.server 8000           # Python 3
npx serve .                          # Node.js
php -S localhost:8000                # PHP

# Open browser
open http://localhost:8000
```

### Controls

- **Click**: Spawn particle bursts
- **Gravity**: Toggle gravitational physics
- **Bloom Effect**: Toggle glow post-processing
- **Particle Trails**: Toggle fading particle trails
- **Friction**: Adjust particle energy decay
