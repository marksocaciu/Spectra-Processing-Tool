#!/usr/bin/env sh
set -eu
# Uncomment the following lines to install Homebrew and required Python packages on macOS
# if ! command -v brew >/dev/null 2>&1; then
#     echo "Homebrew not found. Installing Homebrew..."
#     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#     echo "export PATH=/opt/homebrew/bin:$PATH" >> ~/.bash_profile && source ~/.bash_profile
#     brew install python3
#     brew install python-tk@3.14
# fi
exec python3 "$(dirname "$0")/bootstrap.py" "$@"
