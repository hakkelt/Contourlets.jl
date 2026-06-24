@testitem "Aqua quality checks" tags = [:quality] begin
    using Aqua
    Aqua.test_all(
        Contourlets;
        persistent_tasks = false,  # system resource check can fail in constrained envs
    )
end
