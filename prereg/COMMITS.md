# Pre-registration history

Each file `PREREG_<n>_<hash>.md` is the pre-registration exactly as committed at that time (exported with
`git show <hash>:PREREG.md` from the private working repository). Commit times are git commit times (+06:00).
`PREREG_8f4c029.md.ots` is an OpenTimestamps proof anchoring version 8f4c029 (amendments 1-5) in the Bitcoin
blockchain; verify with `ots verify PREREG_8f4c029.md.ots` against `PREREG_*_8f4c029.md` renamed to
`PREREG_8f4c029.md`. Amendment 6 (afa462e) has no OpenTimestamps anchor of its own; it rests on its commit time.

| # | Commit | Time | Change |
|---|---|---|---|
| 01 | `4444a8a` | 2026-10-20 00:22:38 +0600 | pre-register no-brief test and decision-probe curve |
| 02 | `07ce7b2` | 2026-10-20 00:25:22 +0600 | PREREG amendment 1 (scenario clusters, curve noise, reading) |
| 03 | `b3042d7` | 2026-10-20 10:31:08 +0600 | fix(paper1): learning-curve subset pool; PREREG amendment 2 |
| 04 | `0dc61d7` | 2026-10-20 15:07:28 +0600 | feat(paper1): no-brief main analysis (pre-registered): both probes succeed |
| 05 | `42303c1` | 2026-10-20 16:22:09 +0600 | PREREG amendment 3 (exploratory): text readers and concealed rounds on the no-brief episodes |
| 06 | `fb4b517` | 2026-10-20 16:22:36 +0600 | retitle; fix text contradicted by H5 and the no-brief test |
| 07 | `5e664fa` | 2026-10-20 16:36:35 +0600 | PREREG amendment 4: concealed messages cut at the first coalition advocacy, round-matched |
| 08 | `7686f51` | 2026-10-20 16:43:06 +0600 | PREREG amendment 5: H5 replication on code review and Mistral-24B; h5 --prefix/--name |
| 09 | `8f4c029` | 2026-10-20 16:46:18 +0600 | correct amendment 3 timestamp to its commit time; timestamps note; H5 scenario CIs |
| 10 | `de8c66e` | 2026-10-20 16:54:52 +0600 | PREREG push times and OpenTimestamps proof of the 8f4c029 version |
| 11 | `afa462e` | 2026-10-20 22:31:41 +0600 | PREREG amendment 6: private-cut concealed check, AcMAS-style divergence check, no-model baseline |
| 12 | `d8a77fa` | 2026-10-20 23:10:16 +0600 | PREREG note: run the Qwen3-32B H5 replication (amendment 5 item 4), layers 26/29/32 |
| 13 | `29f5a51` | 2026-10-20 23:18:53 +0600 | queue order note (32B H5 before final-layer extraction); work report |
| 14 | `e200b1c` | 2026-10-20 23:19:01 +0600 | queue order note: time = commit time (23:18) |
| 15 | `a3b4604` | 2026-10-21 23:46:19 +0600 | PREREG note: OpenTimestamps proof of 8f4c029 upgraded to Bitcoin |
| 16 | `c3b0c3b` | 2026-10-25 15:02:36 +0600 | PREREG amendment 7: fix option labels in briefs and regenerate affected episodes |
