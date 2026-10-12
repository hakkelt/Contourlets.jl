# qfb_decompose / qfb_reconstruct overloads for _AbstractGPUMatrix.  The
# shear-based two-channel ladder bank of `src/directional/quincunx_fb.jl` has no
# device kernels yet, so these throw a clear error instead of falling back to the
# CPU loops (scalar indexing).  The DFB and NSDFB run on the device through their
# own FFT banks and do not use the QFB.

function Contourlets.qfb_decompose(image::_AbstractGPUMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    throw(ArgumentError("qfb_decompose is not yet supported on GPU arrays"))
end

function Contourlets.qfb_reconstruct(
        sb0::_AbstractGPUMatrix, sb1::_AbstractGPUMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    throw(ArgumentError("qfb_reconstruct is not yet supported on GPU arrays"))
end
