################################################################################
# check-laksa-mlir
#
# The laksa-mlir regression test project.
################################################################################

project(check-laksa-mlir)

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    laksa-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the laksa-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(DFGMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)
