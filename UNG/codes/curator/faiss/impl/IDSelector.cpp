// Minimal: only MultiTenantIDSelector::is_member implementation.
// IDSelectorRange etc. are provided by NaviX FAISS.

#include "faiss_compat.h"
#include "IDSelector.h"

namespace faiss {

bool MultiTenantIDSelector::is_member(faiss_navix::idx_t id) const {
    FAISS_NAVIX_THROW_IF_NOT_MSG(
        access_map != nullptr, "access_map is required for MultiTenantIDSelector");

    auto it = access_map->find(id);
    if (it == access_map->end()) {
        // Vector doesn't exist
        return false;
    }

    // Check if the tenant has access to this vector
    if (it->second.find(tid) == it->second.end()) {
        return false;
    }

    // If base selector is provided, also check that
    if (base_sel != nullptr) {
        return base_sel->is_member(
            direct_map != nullptr ? direct_map->get(id) : id);
    }

    return true;
}

}  // namespace faiss
