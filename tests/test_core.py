from antenna_core import calculate_antenna, compare_materials


def test_default_point_is_valid_small_loop():
    result = calculate_antenna(5.0, 30.0, 1.0, 1, 50.0, "Đồng (Cu)")
    assert result["Rr_ohm"] > 0
    assert result["Zin_real_ohm"] > 0
    assert result["Zin_imag_ohm"] > 0
    assert result["is_small_loop"] is True


def test_material_comparison_has_four_materials():
    rows = compare_materials(5.0, 30.0, 1.0, 1, 50.0)
    assert len(rows) == 4
    assert len({row["material"] for row in rows}) == 4
    assert all(row["Rac_ohm"] > 0 for row in rows)
