"""Sanity checks on the synthetic dataset generator."""

from no_show.data import generate_dataset


def test_generated_dataset_has_expected_columns():
    df = generate_dataset(n_rows=200)
    expected = {
        "age",
        "gender",
        "appointment_type",
        "days_before_appointment",
        "previous_appointments",
        "previous_no_shows",
        "weekday",
        "appointment_time",
        "reminder_sent",
        "new_patient",
        "no_show",
    }
    assert set(df.columns) == expected


def test_no_show_is_binary():
    df = generate_dataset(n_rows=200)
    assert set(df["no_show"].unique()) <= {0, 1}


def test_previous_no_shows_never_exceeds_previous_appointments():
    df = generate_dataset(n_rows=500)
    assert (df["previous_no_shows"] <= df["previous_appointments"]).all()


def test_no_show_rate_is_realistic():
    """Not near-0% or near-100% — otherwise the classification task is trivial."""
    df = generate_dataset(n_rows=5000)
    rate = df["no_show"].mean()
    assert 0.05 < rate < 0.6


def test_generation_is_reproducible_with_same_seed():
    df1 = generate_dataset(n_rows=100, seed=1)
    df2 = generate_dataset(n_rows=100, seed=1)
    assert df1.equals(df2)
