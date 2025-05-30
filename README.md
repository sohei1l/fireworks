# Fireworks

A realistic fireworks simulator in the browser where you click to launch rockets that explode into spectacular displays. Built with Rust + WebGL + WebAudio to explore WASM in the browser.

## 🎮 Live Demo

[![Demo](https://img.shields.io/badge/Live%20Demo-Fireworks-blue?style=for-the-badge)](https://sohei1l.github.io/fireworks/)

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
git clone https://github.com/sohei1l/fireworks.git
cd fireworks

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

- **Click/Touch**: Launch firework rockets that explode into colorful displays
- **Position matters**: Click lower on screen = higher, bigger fireworks
- **Audio**: Realistic whoosh and explosion sounds
- **Physics**: Gravity affects explosion particles as they fall
