from peakpredict.common import event_maps as em


def test_event_name_known_and_unknown():
    assert em.event_name("40") == "100m"
    assert em.event_name("99999") == "event_99999"


def test_supported_v1_set():
    assert em.is_supported_v1("40")
    assert em.is_supported_v1("70")
    assert not em.is_supported_v1("60")  # 300m not in v1 sprint set


def test_valid_sex():
    assert em.is_valid_sex(1)
    assert em.is_valid_sex(2)
    assert not em.is_valid_sex(0)
    assert not em.is_valid_sex(3)


def test_sprints_are_lower_better():
    assert em.is_lower_better("40")
    assert em.is_lower_better("70")


def test_field_event_is_higher_better(monkeypatch):
    monkeypatch.setitem(em.EVENT_ID_TO_NAME, "LJ", "Long Jump")
    assert not em.is_lower_better("LJ")
