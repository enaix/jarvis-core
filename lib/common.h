#ifndef JSC_COMMON_H
#define JSC_COMMON_H

#include <iostream>
#include <string>
#include <cstdlib>

namespace jsc {

void internal_err_begin() {
    // Print internal error header
    std::cerr << "=========== JSC_INTERNAL_ERROR BEGIN =========== " << std::endl;
}

[[noreturn]] void internal_err(const std::string& error_code, const char* file, int line) {
    std::cerr << "Internal error encountered in jarvis_core, error code : \"" << error_code << "\" at " << file << ":" << line << std::endl;
    std::cerr << "Report this bug to https://github.com/enaix/jarvis-core" << std::endl;
    std::cerr << "============ JSC_INTERNAL_ERROR END ============" << std::endl;
    abort();
}

}

#endif // JSC_COMMON_H
