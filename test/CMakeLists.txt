llvm_canonicalize_cmake_booleans(
  DataflowScheduler_ENABLE_PYTHON_BINDINGS
  DataflowScheduler_BUILD_TOOLS
  BUILD_SHARED_LIBS
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DataflowScheduler_TEST_DEPENDS
  dataflow-scheduler
  dataflow-scheduler-opt
  FileCheck
)

if(DataflowScheduler_ENABLE_PYTHON_BINDINGS)
  list(APPEND DataflowScheduler_TEST_DEPENDS DataflowSchedulerPythonModules)
endif()

add_lit_testsuite(check-dataflow-scheduler 
    "Running the scheduler regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DataflowScheduler_TEST_DEPENDS}
)
set_target_properties(check-dataflow-scheduler PROPERTIES FOLDER "Tests")
add_dependencies(check-all 
  check-dataflow-scheduler 
  check-dataflow-scheduler-dialects
)

add_lit_testsuites(dataflow-scheduler 
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${DataflowScheduler_TEST_DEPENDS}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)
