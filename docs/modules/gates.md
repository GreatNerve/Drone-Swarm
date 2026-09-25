# gates

Eval/demo pass-fail on episode history. Training does not call this for `done`.

Spacing is square-body overlap at the same altitude, not 3D Euclidean. An altitude pass-through is allowed.

Tests 1 and 3 score **standoff-ring** error around the shared target. Four drones cannot occupy one point; stacking on the target is a collision, not a better score.

Public: `evaluate_episode`, `combined_score`, `ring_error`, `standoff_cm`. Caps are 40 / 15 / 50 cm.
