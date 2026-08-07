get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_upstream_c_api_library(MLIRCAPILAKSAConversion
        Conversion.cpp

    DEPENDS
        LAKSAConversionPassesIncGen

    LINK_LIBS PUBLIC
        MLIRCAPIIR
        ${conversion_libs}
)
