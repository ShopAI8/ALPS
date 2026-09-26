
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(UNG_HIDE_STATIC_ARCHIVE_SYMBOLS "-Wl,--exclude-libs,ALL")

add_executable(build_UNG_index build_UNG_index.cpp)
target_link_options(build_UNG_index PRIVATE ${UNG_HIDE_STATIC_ARCHIVE_SYMBOLS})
target_link_libraries(build_UNG_index ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem ${ROARING_LIB} onnxruntime OpenMP::OpenMP_CXX)

add_executable(search_UNG_index search_UNG_index.cpp)
target_link_options(search_UNG_index PRIVATE ${UNG_HIDE_STATIC_ARCHIVE_SYMBOLS})
target_link_libraries(search_UNG_index ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem ${ROARING_LIB} onnxruntime OpenMP::OpenMP_CXX)

add_executable(filtered_scan filtered_scan.cpp)
target_link_options(filtered_scan PRIVATE ${UNG_HIDE_STATIC_ARCHIVE_SYMBOLS})
target_link_libraries(filtered_scan ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem ${ROARING_LIB} onnxruntime OpenMP::OpenMP_CXX)

add_executable(analyze_data analyze_data.cpp)
target_link_options(analyze_data PRIVATE ${UNG_HIDE_STATIC_ARCHIVE_SYMBOLS})
target_link_libraries(analyze_data ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem ${ROARING_LIB} onnxruntime OpenMP::OpenMP_CXX)
