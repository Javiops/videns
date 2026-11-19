from app import regions


def test_mask_loads_and_camps_validate():
    # chequeo heredado de build_regions: los 12 campamentos caen en su cuadrante
    assert regions.validate_camps() == []


def test_known_points():
    assert regions.region(0.03, 0.03) == "BASE"
    assert regions.region(0.97, 0.97) == "BASE_ENEMIGA"
    # gromp azul es jungla propia azul; su espejo, jungla enemiga azul
    assert regions.region(0.142, 0.568) == "JGL_PA"
    assert regions.region(1 - 0.142, 1 - 0.568) == "JGL_EA"


def test_mirror_for_red_team():
    # el gromp del equipo rojo es SU jungla propia azul
    assert regions.region_for_team(1 - 0.142, 1 - 0.568, 200) == "JGL_PA"
    assert regions.region_for_team(0.03, 0.03, 200) == "BASE_ENEMIGA"
    # los carriles no se espejan de nombre
    for nx, ny, team in ((0.5, 0.52, 100), (0.5, 0.52, 200)):
        assert regions.region_for_team(nx, ny, team) == "MID"
