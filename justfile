# vectorizeo: build, test and install the CLI and GUI globally (~/.cargo/bin) with shell completions.

shell := file_name(env("SHELL", "bash"))

default:
    @just --list

build:
    cargo build --release

test:
    cargo test

# Run the full pipeline: just trace in.png out.svg -t photo -q high
trace input output *args:
    cargo run --release -q -p vectorizeo -- trace {{input}} {{output}} {{args}}

# Run the desktop GUI: just gui [image]
gui *args:
    cargo run --release -q -p vectorizeo-gui -- {{args}}

# Serve the GUI as a web app on http://127.0.0.1:8080 (needs trunk and the wasm32 target: just web-setup)
web:
    cd crates/vectorizeo-gui && trunk serve --release --open

# Build the web app into crates/vectorizeo-gui/dist (static files; host them anywhere)
web-build:
    cd crates/vectorizeo-gui && trunk build --release

# Build the web app and upload it to Cloudflare Pages (first time: npx wrangler login)
deploy project="vectorizeo": web-build
    npx --yes wrangler pages deploy crates/vectorizeo-gui/dist --project-name {{project}} --commit-dirty=true

# Install what the web build needs
web-setup:
    rustup target add wasm32-unknown-unknown
    cargo binstall -y trunk || cargo install --locked trunk

# Install `vectorizeo` and `vectorizeo-gui` into ~/.cargo/bin
install:
    cargo install --path crates/vectorizeo-cli --locked --force
    cargo install --path crates/vectorizeo-gui --locked --force

# Install completions for zsh, bash or fish (default: $SHELL)
completions sh=shell:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{sh}}" in
      zsh)
        mkdir -p ~/.zfunc
        vectorizeo completions zsh > ~/.zfunc/_vectorizeo
        if ! grep -qs 'fpath+=~/.zfunc' ~/.zshrc; then
          printf '\n# vectorizeo completions\nfpath+=~/.zfunc\nautoload -Uz compinit && compinit\n' >> ~/.zshrc
          echo "added ~/.zfunc to fpath in ~/.zshrc"
        fi
        rm -f ~/.zcompdump*
        echo "installed ~/.zfunc/_vectorizeo (open a new shell)" ;;
      bash)
        dir="${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions"
        mkdir -p "$dir" && vectorizeo completions bash > "$dir/vectorizeo"
        echo "installed $dir/vectorizeo" ;;
      fish)
        dir="${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions"
        mkdir -p "$dir" && vectorizeo completions fish > "$dir/vectorizeo.fish"
        echo "installed $dir/vectorizeo.fish" ;;
      *) echo "unsupported shell: {{sh}}" >&2; exit 1 ;;
    esac

# Install the binary and completions
setup: install completions

uninstall:
    -cargo uninstall vectorizeo
    -cargo uninstall vectorizeo-gui
    rm -f ~/.zfunc/_vectorizeo "${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/vectorizeo" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions/vectorizeo.fish"
