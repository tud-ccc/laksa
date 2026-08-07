################################################################################
# DFGIncGen
#
# MLIR DFG dialect generated includes.
################################################################################

add_custom_target(DFGIncGen)

mlir_gen_enums(DFG)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)
