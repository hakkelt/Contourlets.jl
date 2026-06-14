using GPUEnv

# Build (and activate) a GPU overlay on top of this test environment.  GPUEnv
# adds the GPU stack — GPUArrays, KernelAbstractions, JLArrays and any functional
# native backend (CUDA, AMDGPU, …) — so they need not be listed as test deps and
# the `:gpu` test items can load the ContourletsGPUExt extension.
GPUEnv.activate(persist = true, include_jlarrays = true)

using TestItemRunner
@run_package_tests
