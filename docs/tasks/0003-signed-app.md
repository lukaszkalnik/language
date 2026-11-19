# 0003 Signed and notarized app

Status: concept

## Problem
The released app is not signed: macOS Gatekeeper blocks the first start (System Settings →
Privacy & Security → Open Anyway) and Windows SmartScreen warns. Hard for non-technical users.

## Proposal
Sign and notarize the macOS app and sign the Windows executable in the release workflow.

## Details
- macOS: Apple Developer ID ($99/year), `codesign` with hardened runtime and entitlements for
  the bundled Python, `notarytool`, stapling the `.dmg`; secrets in GitHub Actions.
- Windows: a code-signing certificate (or Azure Trusted Signing), `signtool` in the workflow.
- Remove the workaround steps from `tools/gui/packaging/RELEASE_NOTES.md`.

## Done when
A downloaded release opens with a double-click on a fresh Mac and a fresh Windows machine.
