@testitem "Aqua quality checks" begin
    using Aqua, Contourlets
    Aqua.test_all(
        Contourlets;
        ambiguities = false,       # allow minor method ambiguities from duck-typed dispatch
        stale_deps = false,        # avoid false positives from test-only deps
        persistent_tasks = false
    )  # system resource check can fail in constrained envs
end
