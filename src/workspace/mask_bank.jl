# Where the outputs of MultiscaleFilterBanks' mask-bank engine (`_MaskBankEngine`,
# which runs the NSCT and TICT workspaces) live in this package's coefficient
# sets: `_MaskItem.index` is (level, direction) for the NSCT and
# (level, channel, direction) for the TICT.

MultiscaleFilterBanks._subband(c::NSCTCoefficients, i::NTuple{2, Int}) = c.subbands[i[1]][i[2]]
MultiscaleFilterBanks._subband(c::TICTCoefficients, i::NTuple{3, Int}) = c.subbands[i[1]][i[2]][i[3]]
