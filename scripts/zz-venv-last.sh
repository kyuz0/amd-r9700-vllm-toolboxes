#!/usr/bin/env bash
# Ensure /opt/venv/bin is first even if ~/.local/bin or ~/.cargo/bin prepend themselves via user dotfiles.

_venv_path_fix() {
  # remove any existing /opt/venv/bin entries, then prepend one
  local newpath
  newpath="$(printf '%s' "$PATH" | awk -v RS=: -v ORS=: '$0!="/opt/venv/bin"{print}')"
  PATH="/opt/venv/bin:${newpath%:}"
}

# run once after shell init; don't duplicate
case "$PROMPT_COMMAND" in
  *_venv_path_fix*) : ;;
  *) PROMPT_COMMAND="_venv_path_fix${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;;
esac

