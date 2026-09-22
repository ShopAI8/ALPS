set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES utils.cpp storage.cpp trie.cpp distance.cpp search_queue.cpp filtered_scan.cpp uni_nav_graph.cpp MethodSelector.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glibc_compat.cpp")
    list(APPEND CPP_SOURCES glibc_compat.cpp)
else()
    message(STATUS "glibc_compat.cpp not found; building without glibc compatibility shim")
endif()
if(ENABLE_RABITQ)
    list(APPEND CPP_SOURCES rabitq_side_index.cpp)
endif()

# Curator sources — compiled directly into libANNS
list(APPEND CPP_SOURCES
    ../curator/curator_wrapper.cpp
    ../curator/faiss/MultiTenantIndex.cpp
    ../curator/faiss/MultiTenantIndexIVF.cpp
    ../curator/faiss/MultiTenantIndexIVFFlat.cpp
    ../curator/faiss/MultiTenantIndexIVFHierarchical.cpp
    ../curator/faiss/complex_predicate.cpp
    ../curator/faiss/impl/IDSelector.cpp
)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE faiss ${ROARING_LIB})
if(ENABLE_KNOWHERE_MILVUS_BASELINE)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${KNOWHERE_LIBRARY})
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../FAVOR/include)

# Curator include paths (PRIVATE — only for Curator .cpp files, not propagated)
# curator/            → for curator_wrapper.h
# curator/faiss/      → for local Curator headers (MultiTenantIndex.h etc.)
# NaviX/              → for <faiss_navix/XXX.h> (NaviX FAISS headers)
# NaviX/faiss_navix/  → for NaviX internal includes (<faiss_navix/YYY.h>)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../curator
    ${CMAKE_CURRENT_SOURCE_DIR}/../curator/faiss
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NaviX
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NaviX/faiss_navix
)

if(URING_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${URING_LIB})
endif()
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)
endif()
