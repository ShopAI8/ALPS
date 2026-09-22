// Minimal Curator extension to NaviX's IDSelector: adds MultiTenantIDSelector.
// Does NOT redefine IDSelector — uses faiss_navix::IDSelector via compat layer.

#pragma once

#include <faiss_compat.h>                    // namespace bridge + all NaviX headers
#include <faiss_navix/invlists/DirectMap.h>  // DirectMap (not in compat)

namespace faiss {

/** Curator-specific: filters vectors by tenant ID AND a base selector. */
struct MultiTenantIDSelector : faiss_navix::IDSelector {
    tid_t tid;
    const AccessMap* access_map;
    const faiss_navix::DirectMap* direct_map;
    const faiss_navix::IDSelector* base_sel;

    MultiTenantIDSelector(
        tid_t tid,
        const AccessMap* access_map,
        const faiss_navix::DirectMap* direct_map,
        const faiss_navix::IDSelector* base_sel)
        : tid(tid), access_map(access_map),
          direct_map(direct_map), base_sel(base_sel) {}

    bool is_member(faiss_navix::idx_t id) const override;
};

}  // namespace faiss
