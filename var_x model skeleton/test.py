
# ---------------- TEMPORARY SMOKE TEST (delete once real data flows) ----------------
from varx_skeleton import plot_series


if __name__ == "__main__":
    # 1. Fabricate a dataset and write it to CSV (tests the load_data round-trip)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    fake = pd.DataFrame(
        rng.standard_normal((200, 4)).cumsum(axis=0),
        index=idx,
        columns=["y1", "y2", "x1", "x2"],
    )
    fake.index.name = "Date"
    fake.sample(frac=1, random_state=0).to_csv("_smoke_test.csv")  # shuffled on purpose

    # 2. load_data
    df = load_data("_smoke_test.csv")
    assert df.index.is_monotonic_increasing, "sort_index failed"
    assert df.shape == (200, 4), f"unexpected shape {df.shape}"
    print("load_data OK:", df.shape)

    # 3. split_endog_exog — temporarily fill config for the test
    ENDOG_COLS[:] = ["y1", "y2"]
    EXOG_COLS[:] = ["x1", "x2"]
    endog, exog = split_endog_exog(df)
    print("split OK:", endog.columns.tolist(), "|", exog.columns.tolist())

    # 4. Confirm the guard clause actually fires
    ENDOG_COLS[:] = ["y1", "typo_col"]
    try:
        split_endog_exog(df)
        print("GUARD FAILED - no error raised")
    except KeyError as e:
        print("guard OK:", e)
    ENDOG_COLS[:] = ["y1", "y2"]

    # 5. plot_series
    fig = plot_series(endog, save_path="_smoke_plot.png")
    print("plot OK:", len(fig.axes), "axes; saved _smoke_plot.png")

    print("\nAll smoke tests passed.")