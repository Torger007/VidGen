import builtins


def test_load_openpose_detector_returns_none_when_dependency_import_crashes(monkeypatch) -> None:
    import app.services.condition_preprocessors as condition_preprocessors

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"controlnet_aux", "controlnet_aux.open_pose"}:
            raise AttributeError("module 'mediapipe' has no attribute 'solutions'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    condition_preprocessors.clear_preprocessor_caches()

    assert condition_preprocessors._load_openpose_detector() is None


def test_load_openpose_detector_falls_back_to_open_pose_module(monkeypatch) -> None:
    import app.services.condition_preprocessors as condition_preprocessors

    class FakeDetector:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "controlnet_aux":
            raise AttributeError("module 'mediapipe' has no attribute 'solutions'")
        if name == "controlnet_aux.open_pose":
            module = type("FakeModule", (), {"OpenposeDetector": FakeDetector})
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    condition_preprocessors.clear_preprocessor_caches()

    detector = condition_preprocessors._load_openpose_detector()
    assert isinstance(detector, FakeDetector)


def test_load_depth_estimator_returns_none_when_transformers_import_crashes(monkeypatch) -> None:
    import app.services.condition_preprocessors as condition_preprocessors

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            raise RuntimeError("broken transformers import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    condition_preprocessors.clear_preprocessor_caches()

    assert condition_preprocessors._load_depth_estimator() is None
