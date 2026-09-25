# tasks

Test 1 static shared_target, Test 2 geometry slots, Test 3 moving shared_target (scripted 3D Lissajous; joystick override).

All three tests accept `set_joystick_target` for the live demo (Test 2 translates the geometry origin). Test 2 also accepts `set_kind(square|rectangle|circle)` and rebuilds slots immediately.

Public: `make_task(test_id, cfg) -> Task`. Depends on numpy.
