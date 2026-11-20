@testitem "Aqua quality checks" tags = [:quality] begin
    using Aqua
    Aqua.test_all(
        Contourlets;
        persistent_tasks = false,  # system resource check can fail in constrained envs
    )
end

@testitem "ExplicitImports" tags = [:quality] begin
    using ExplicitImports
    # Public-ness is only known from Julia 1.11 (`Base.ispublic`); older versions
    # report every non-exported name as non-public.
    public_checks = VERSION >= v"1.11"
    # Loaded package extensions are checked too; they may use Contourlets's internals.
    internal = public_checks ? Tuple(n for n in names(Contourlets; all = true) if !Base.ispublic(Contourlets, n)) : ()
    test_explicit_imports(
        Contourlets;
        # The GPU extension reaches `Adapt` through KernelAbstractions: Adapt is not a
        # trigger of the extension, so it cannot be loaded directly.
        all_explicit_imports_via_owners = (; ignore = (:Adapt,)),
        all_explicit_imports_are_public = public_checks ? (; ignore = (:Adapt, internal...)) : false,
        all_qualified_accesses_are_public = public_checks ? (; ignore = (:ESTIMATE, :MEASURE, :UNALIGNED, internal...)) : false,
    )
end
