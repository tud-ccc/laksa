################################################################################
# ladle
#
# The central LAKSA compiler driver: dispatches to laksa-opt and
# laksa-translate to turn an input file into an output file.
################################################################################

project(ladle)

add_executable(${PROJECT_NAME}
    ladle.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        LLVMSupport
)

add_dependencies(${PROJECT_NAME} laksa-opt laksa-translate)
