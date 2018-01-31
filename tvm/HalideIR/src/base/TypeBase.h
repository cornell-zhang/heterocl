#ifndef HALIDEIR_TYPEBASE_H
#define HALIDEIR_TYPEBASE_H

// type handling code stripped from Halide runtime

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
// Forward declare type to allow naming typed handles.
// See Type.h for documentation.
template<typename T> struct halideir_handle_traits;

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats (of various bit-widths), or a handle (which is always 64-bits).
 * Note that the int/uint/float values do not imply a specific bit width
 * (the bit width is expected to be encoded in a separate value).
 */
typedef enum halideir_type_code_t
#if __cplusplus >= 201103L
: uint8_t
#endif
{
    halideir_type_int = 0,   //!< signed integers
    halideir_type_uint = 1,  //!< unsigned integers
    halideir_type_float = 2, //!< floating point numbers
    halideir_type_handle = 3 //!< opaque pointer type (void *)
} halideir_type_code_t;

// Note that while __attribute__ can go before or after the declaration,
// __declspec apparently is only allowed before.
#ifndef HALIDEIR_ATTRIBUTE_ALIGN
    #ifdef _MSC_VER
        #define HALIDEIR_ATTRIBUTE_ALIGN(x) __declspec(align(x))
    #else
        #define HALIDEIR_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
    #endif
#endif

/** A runtime tag for a type in the halide type system. Can be ints,
 * unsigned ints, or floats of various bit-widths (the 'bits'
 * field). Can also be vectors of the same (by setting the 'lanes'
 * field to something larger than one). This struct should be
 * exactly 32-bits in size. */
struct halideir_type_t {
    /** The basic type code: signed integer, unsigned integer, or floating point. */
#if __cplusplus >= 201103L
    HALIDEIR_ATTRIBUTE_ALIGN(1) halideir_type_code_t code; // halideir_type_code_t
#else
    HALIDEIR_ATTRIBUTE_ALIGN(1) uint8_t code; // halideir_type_code_t
#endif

    /** The number of bits of precision of a single scalar value of this type. */
    HALIDEIR_ATTRIBUTE_ALIGN(1) uint8_t bits;

    /** How many elements in a vector. This is 1 for scalar types. */
    HALIDEIR_ATTRIBUTE_ALIGN(2) uint16_t lanes;

#ifdef __cplusplus
    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    halideir_type_t(halideir_type_code_t code, uint8_t bits, uint16_t lanes = 1)
        : code(code), bits(bits), lanes(lanes) {
    }

    /** Default constructor is required e.g. to declare halideir_trace_event
     * instances. */
    halideir_type_t() : code((halideir_type_code_t)0), bits(0), lanes(0) {}

    /** Compare two types for equality. */
    bool operator==(const halideir_type_t &other) const {
        return (code == other.code &&
                bits == other.bits &&
                lanes == other.lanes);
    }

    /** Size in bytes for a single element, even if width is not 1, of this type. */
    size_t bytes() const { return (bits + 7) / 8; }
#endif
};

namespace {

template<typename T>
struct halideir_type_of_helper;

template<typename T>
struct halideir_type_of_helper<T *> {
    operator halideir_type_t() {
        return halideir_type_t(halideir_type_handle, 64);
    }
};

template<typename T>
struct halideir_type_of_helper<T &> {
    operator halideir_type_t() {
        return halideir_type_t(halideir_type_handle, 64);
    }
};

// Halide runtime does not require C++11
#if __cplusplus > 199711L
template<typename T>
struct halideir_type_of_helper<T &&> {
    operator halideir_type_t() {
        return halideir_type_t(halideir_type_handle, 64);
    }
};
#endif

template<>
struct halideir_type_of_helper<float> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_float, 32); }
};

template<>
struct halideir_type_of_helper<double> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_float, 64); }
};

template<>
struct halideir_type_of_helper<uint8_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_uint, 8); }
};

template<>
struct halideir_type_of_helper<uint16_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_uint, 16); }
};

template<>
struct halideir_type_of_helper<uint32_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_uint, 32); }
};

template<>
struct halideir_type_of_helper<uint64_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_uint, 64); }
};

template<>
struct halideir_type_of_helper<int8_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_int, 8); }
};

template<>
struct halideir_type_of_helper<int16_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_int, 16); }
};

template<>
struct halideir_type_of_helper<int32_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_int, 32); }
};

template<>
struct halideir_type_of_helper<int64_t> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_int, 64); }
};

template<>
struct halideir_type_of_helper<bool> {
    operator halideir_type_t() { return halideir_type_t(halideir_type_uint, 1); }
};

}

/** Construct the halide equivalent of a C type */
template<typename T> halideir_type_t halideir_type_of() {
    return halideir_type_of_helper<T>();
}

// it is not necessary, and may produce warnings for some build configurations.
#ifdef _MSC_VER
#define HALIDEIR_ALWAYS_INLINE __forceinline
#else
#define HALIDEIR_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

#endif // HALIDEIR_HALIDERUNTIME_H
