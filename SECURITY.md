# Security Policy

## Supported versions

Harmonia is released from `main` and versioned with the dataset/spec it ships
(currently **0.7.x**). Security and data-integrity fixes are applied to the
latest tagged release and to `main`. Older tags are not patched in place — pin a
release for reproducibility, but upgrade to pick up fixes.

| Version | Supported |
| --- | --- |
| latest `0.7.x` / `main` | ✅ |
| earlier tags | ❌ (upgrade to the latest release) |

## Scope — read this first

Harmonia is **scientific infrastructure, not a clinical or regulatory tool**. It
emits torsade-risk *distributions* and classification-*flip frequencies*, never a
bare safety verdict. A model output you disagree with, a tier you would assign
differently, or a curated parameter you believe is wrong is **not a security
issue** — it is a dataset or science question. Please open a normal
[GitHub issue](https://github.com/clay-good/harmonia/issues) (or a pull request
following [CONTRIBUTING.md](CONTRIBUTING.md)) for those, including the primary
source you are checking against.

Report **privately** (below) only for genuine security vulnerabilities, such as:

- Code execution, path traversal, or unsafe deserialization reachable through
  the package, CLI, exporters, or the Streamlit dashboard
- A crafted dataset record, citation, or export file that can execute code or
  exfiltrate data when validated, loaded, or rendered
- Dependency vulnerabilities that are actually exploitable through Harmonia's
  use of them
- Any defect whose **public** disclosure would put users at risk before a fix
  exists

## Reporting a vulnerability

Please do **not** open a public issue for a security vulnerability.

1. Preferred: use GitHub's
   [private vulnerability reporting](https://github.com/clay-good/harmonia/security/advisories/new)
   ("Report a vulnerability" on the repository's **Security** tab).
2. Or email **hi@claygood.com** with a clear subject line beginning
   `SECURITY:` and enough detail to reproduce.

Please include:

- The affected component and version (or commit SHA)
- A minimal reproduction or proof of concept
- The impact you observed and any suggested remediation

## What to expect

- **Acknowledgement** within 5 business days.
- An initial assessment (severity, affected versions) within 10 business days.
- Coordinated disclosure: we will agree on a timeline with you, fix on `main`,
  cut a patched release, and credit you in the release notes unless you prefer
  to remain anonymous.

Thank you for helping keep Harmonia and its users safe.
