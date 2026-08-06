No custom fixture image is committed here. `test_person_detector.py` uses the `bus.jpg` sample
image bundled with the `ultralytics` package itself (`ultralytics.utils.ASSETS`) — it contains
real pedestrians, needs no network access beyond what installing `ultralytics` already requires,
and avoids committing a binary to the repo.
