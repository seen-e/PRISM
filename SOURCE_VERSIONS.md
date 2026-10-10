# Bundled source snapshots

- `openpi/` was copied from the authors' active OpenPI/π0.5 working tree on
  2026-10-08. The enclosing project revision was
  `a010c7d0bfa13dfe6aff42fdebc7677c32cbb9bb`; the snapshot also includes the
  working-tree PRISM changes used by the current experiments.
- `starvla/` is based on StarVLA commit
  `985c2238627481f25476fbee478083812dd82a21` with the repository's PRISM
  QwenPI-v3/LayerwiseFM integration applied directly to the bundled source.

Neither implementation requires downloading or patching another source
repository. Third-party Python packages and model initializations remain
external dependencies.
