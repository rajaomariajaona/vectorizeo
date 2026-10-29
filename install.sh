#!/usr/bin/env bash
# Install vectorizeo (CLI + GUI) from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/rajaomariajaona/vectorizeo/main/install.sh | bash
#
# Options (environment variables):
#   VECTORIZEO_VERSION  release tag to install, e.g. v0.1.0 (default: latest)
#   VECTORIZEO_PREFIX   install root (default: ~/.local) -> bin/, share/applications/, share/icons/
# Uninstall:  curl -fsSL .../install.sh | bash -s -- --uninstall
set -euo pipefail

REPO="rajaomariajaona/vectorizeo"
PREFIX="${VECTORIZEO_PREFIX:-$HOME/.local}"
VERSION="${VECTORIZEO_VERSION:-latest}"
BIN="$PREFIX/bin"
APPS="$PREFIX/share/applications"
ICONS="$PREFIX/share/icons/hicolor/scalable/apps"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$BIN/vectorizeo" "$BIN/vectorizeo-gui" "$APPS/vectorizeo.desktop" "$ICONS/vectorizeo.svg"
  rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/vectorizeo" \
        "${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions/vectorizeo.fish" \
        "$HOME/.zfunc/_vectorizeo"
  say "vectorizeo removed"
  exit 0
fi

[ "$(uname -s)" = "Linux" ] || die "this installer is for Linux; on Windows download the zip from https://github.com/$REPO/releases"
case "$(uname -m)" in
  x86_64|amd64) arch=x86_64 ;;
  *) die "no prebuilt package for $(uname -m); build from source with: cargo install --git https://github.com/$REPO vectorizeo" ;;
esac

if command -v curl >/dev/null; then dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null; then dl() { wget -qO "$2" "$1"; }
else die "curl or wget is required"; fi

name="vectorizeo-$arch-linux"
if [ "$VERSION" = "latest" ]; then
  base="https://github.com/$REPO/releases/latest/download"
else
  base="https://github.com/$REPO/releases/download/$VERSION"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

say "downloading $name ($VERSION)"
dl "$base/$name.tar.gz" "$tmp/$name.tar.gz" || die "download failed: $base/$name.tar.gz"

if dl "$base/SHA256SUMS" "$tmp/SHA256SUMS" 2>/dev/null && command -v sha256sum >/dev/null; then
  (cd "$tmp" && grep " $name.tar.gz\$" SHA256SUMS | sha256sum -c - >/dev/null) || die "checksum mismatch"
  say "checksum ok"
fi

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
src="$tmp/$name"

say "installing to $PREFIX"
mkdir -p "$BIN" "$APPS" "$ICONS"
install -m 755 "$src/vectorizeo" "$src/vectorizeo-gui" "$BIN/"
install -m 644 "$src/vectorizeo.svg" "$ICONS/vectorizeo.svg"
cat > "$APPS/vectorizeo.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Vectorizeo
Comment=Turn bitmap images into vector graphics
Exec=$BIN/vectorizeo-gui %f
Icon=vectorizeo
Categories=Graphics;
MimeType=image/png;image/jpeg;image/bmp;image/gif;image/tiff;
StartupWMClass=vectorizeo-gui
Terminal=false
DESKTOP
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true

# Shell completions for the user's shell, if it is one we know.
case "$(basename "${SHELL:-}")" in
  bash)
    d="${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions"
    mkdir -p "$d" && "$BIN/vectorizeo" completions bash > "$d/vectorizeo" ;;
  fish)
    d="${XDG_CONFIG_HOME:-$HOME/.config}/fish/completions"
    mkdir -p "$d" && "$BIN/vectorizeo" completions fish > "$d/vectorizeo.fish" ;;
  zsh)
    mkdir -p "$HOME/.zfunc" && "$BIN/vectorizeo" completions zsh > "$HOME/.zfunc/_vectorizeo"
    say "zsh: make sure 'fpath+=~/.zfunc' comes before compinit in ~/.zshrc" ;;
esac

say "installed: $("$BIN/vectorizeo" --version 2>/dev/null || echo vectorizeo)"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) say "add $BIN to your PATH:  export PATH=\"$BIN:\$PATH\"" ;;
esac
