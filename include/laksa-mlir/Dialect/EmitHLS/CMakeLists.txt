################################################################################
# EmitHLSIncGen
#
# MLIR EmitHLS dialect generated includes.
################################################################################

add_custom_target(EmitHLSIncGen)

mlir_gen_enums(EmitHLS)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)
