
# include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(fvecs_to_bin fvecs_to_bin.cpp)
target_link_libraries(fvecs_to_bin
    PRIVATE
    ${Boost_LIBRARIES}
    ${PROJECT_NAME}
    ${ROARING_LIB}
    onnxruntime
    faiss
    ${NAVIX_LIB}
)

add_executable(generate_base_labels generate_base_labels.cpp)
target_link_libraries(generate_base_labels
    PRIVATE
    ${Boost_LIBRARIES}
    ${PROJECT_NAME}
    ${ROARING_LIB}
    onnxruntime
    faiss
    ${NAVIX_LIB}
)

add_executable(generate_query_labels generate_query_labels.cpp)
target_link_libraries(generate_query_labels
    PRIVATE
    ${Boost_LIBRARIES}
    ${PROJECT_NAME}
    ${ROARING_LIB}
    onnxruntime
    faiss
    ${NAVIX_LIB}
)

add_executable(compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth
    PRIVATE
    ${Boost_LIBRARIES}
    ${PROJECT_NAME}
    ${ROARING_LIB}
    onnxruntime
    faiss
    ${NAVIX_LIB}
)

add_executable(generate_mixed_queries generate_mixed_queries.cpp)
target_link_libraries(generate_mixed_queries
    PRIVATE
    ${Boost_LIBRARIES}
    ${PROJECT_NAME}
    ${ROARING_LIB}
    onnxruntime
    faiss
    ${NAVIX_LIB}
)
