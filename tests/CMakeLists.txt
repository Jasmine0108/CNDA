include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(catch2)

# C++ Core Tests
add_executable(test_core
    cpp/core/test_sanity.cpp 
    cpp/core/test_dtypes.cpp 
    cpp/core/test_view.cpp
)
target_link_libraries(test_core PRIVATE Catch2::Catch2WithMain cnda_headers)

# C++ AoS Tests
add_executable(test_aos
    cpp/aos/test_basic.cpp 
    cpp/aos/test_field_layout.cpp 
    cpp/aos/test_indexing.cpp
)
target_link_libraries(test_aos PRIVATE Catch2::Catch2WithMain cnda_headers)

# 讓 CTest 能發現並執行
include(Catch)
catch_discover_tests(test_core)
catch_discover_tests(test_aos)