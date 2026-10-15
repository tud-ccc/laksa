find_package(Doxygen)

if (DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # Prefer the configured Python's Sphinx over the one on PATH.
    if (Python3_EXECUTABLE)
        set(SPHINX_CMD ${Python3_EXECUTABLE} -m sphinx)
        set(SPHINX_INSTALL_DEPS
            COMMAND ${Python3_EXECUTABLE} -m pip install
                    -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt)
        message(STATUS "Sphinx will run via: ${Python3_EXECUTABLE} -m sphinx")
    else()
        find_program(SPHINX_BUILD_EXE sphinx-build)
        if (SPHINX_BUILD_EXE)
            set(SPHINX_CMD ${SPHINX_BUILD_EXE})
            message(STATUS "Sphinx will run via: ${SPHINX_BUILD_EXE}")
        endif()
    endif()

    if (SPHINX_CMD)
        # Target of docs links to source files.
        set(LAKSA_DOCS_REPO_URL "https://github.com/tud-ccc/laksa"
            CACHE STRING "Repository URL that docs links to source files use")
        set(LAKSA_DOCS_REPO_REF "main"
            CACHE STRING "Branch, tag or commit that docs links to source files use")

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
            ${CMAKE_CURRENT_BINARY_DIR}/conf.py
            @ONLY)

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
            ${CMAKE_CURRENT_BINARY_DIR}/index.rst
            COPYONLY)

        configure_file(
            ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_BINARY_DIR}/README.md
            COPYONLY)

        # Served at docs/LAKSA-{Dark,Light}.png, the paths README.md uses.
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Dark.png
            ${CMAKE_CURRENT_BINARY_DIR}/docs/docs/LAKSA-Dark.png
            COPYONLY)

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Light.png
            ${CMAKE_CURRENT_BINARY_DIR}/docs/docs/LAKSA-Light.png
            COPYONLY)

        # html_logo is resolved relative to conf.py.
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Dark.png
            ${CMAKE_CURRENT_BINARY_DIR}/LAKSA-Dark.png
            COPYONLY)

        set(SPHINX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

        add_custom_target(build-laksa-doc
            ${SPHINX_INSTALL_DEPS}
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            # Drop pages of symbols that no longer exist.
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/api
            COMMAND ${SPHINX_CMD}
                    -b html
                    # Pick up conf.py changes.
                    -E
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${SPHINX_OUTPUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation (Doxygen XML + Sphinx HTML)"
            VERBATIM)

        message(STATUS
            "Run 'cmake --build . --target build-laksa-doc' to build docs")
    else()
        add_custom_target(build-laksa-doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating Doxygen XML (sphinx not found)"
            VERBATIM)

        message(STATUS
            "sphinx not found — set Python3_EXECUTABLE or install sphinx-build")
    endif()

    add_dependencies(build-laksa-doc mlir-doc)
endif()
