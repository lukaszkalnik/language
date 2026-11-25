# 0003 Signed and notarized app

Status: parked

**Parked** by the user on 2026-11-25: no paid Apple Developer Program membership for now, and no
free way to sign the Windows executable yet (see Findings). Revisit when one of the routes below
becomes worth it.

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

## Findings (2026-11-25)
- macOS: needs the Apple Developer Program ($99/year); not now.
- Windows, paid: Azure Trusted Signing (about $10/month) is not open to individual developers in
  Germany; a certificate from a CA costs about $200–400/year and lives on a token or cloud HSM.
- Windows, free: **SignPath Foundation** signs open-source projects for free, but this repository
  has no licence yet (an OSI licence is required), the publisher shown is "SignPath Foundation",
  and releases must credit it. The **Microsoft Store** re-signs MSIX packages for free (the route
  the user's other app, Sito, plans), but the store container redirects file writes, which needs
  testing with the first-start install of 2–3 GB of tools and the model downloads; the GitHub
  download would stay unsigned.

## Done when
A downloaded release opens with a double-click on a fresh Mac and a fresh Windows machine.
