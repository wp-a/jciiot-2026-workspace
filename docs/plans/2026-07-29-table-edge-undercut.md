# L1 Table-Edge Undercut Implementation Plan

1. Add failing tests for an opt-in no-grasp runner mode, conservative parser
   defaults, and measured support acceptance.
2. Implement a table-edge undercut probe with staged open-gripper Cartesian
   targets and collision/contact logging.
3. Run the focused tests, full project suite, scored-path audit, and syntax
   checks.
4. Sync only the external runner to the pinned server and verify that the 8502
   process remains on its independent candidate.
5. Run one L1 physical trial and archive the compact result locally.
6. Record the measured result before selecting the next single variable.
