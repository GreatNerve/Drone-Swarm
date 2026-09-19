# tasks

Test 1 static shared_target, Test 2 geometry slots, Test 3 moving shared_target (joystick override).

Test 1 and Test 3 accept `set_joystick_target` for the live demo. Test 2 accepts `set_kind(square|rectangle|circle)` and rebuilds slots immediately.

Public: `make_task(test_id, cfg) -> Task`. Depends on numpy.
