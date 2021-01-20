#ifndef HALIDEIR_TYPE_H
#define HALIDEIR_TYPE_H

#include <stdint.h>
#include "TypeBase.h"
#include "Error.h"
#include "Util.h"
#include "Float16.h"

/** \file
 * Defines halide types
 */

/** A set of types to represent a C++ function signature. This allows
 * two things.  First, proper prototypes can be provided for Halide
 * generated functions, giving better compile time type
 * checking. Second, C++ name mangling can be done to provide link
 * time type checking for both Halide generated functions and calls
 * from Halide to external functions.
 *
 * These are intended to be constexpr producable, but we don't depend
 * on C++11 yet. In C++14, it is possible these will be replaced with
 * introspection/reflection facilities.
 *
 * halideir_handle_traits has to go outside the Halide namespace due to template
 * resolution rules. TODO(zalman): Do all types need to be in global namespace?
 */
 //@{

/** A structure to represent the (unscoped) name of a C++ composite type for use
 * as a single argument (or return value) in a function signature.
 *
 * Currently does not support the restrict qualifier, references, or
 * r-value references.  These features cannot be used in extern
 * function calls from Halide or in the generated function from
 * Halide, but their applicability seems limited anyway.
 */
struct halideir_cplusplus_type_name {
    /// An enum to indicate whether a C++ type is non-composite, a struct, class, or union
    enum CPPTypeType {
        Simple, ///< "int"
        Struct, ///< "struct Foo"
        Class,  ///< "class Foo"
        Union,  ///< "union Foo"
        Enum,   ///< "enum Foo"
    } cpp_type_type;  // Note: order is reflected in map_to_name table in CPlusPlusMangle.cpp

    std::string name;

    halideir_cplusplus_type_name(CPPTypeType cpp_type_type, const std::string &name)
        : cpp_type_type(cpp_type_type), name(name) {
    }

    bool operator==(const halideir_cplusplus_type_name &rhs) const {
         return cpp_type_type == rhs.cpp_type_type &&
                name == rhs.name;
    }

    bool operator!=(const halideir_cplusplus_type_name &rhs) const {
        return !(*this == rhs);
    }

    bool operator<(const halideir_cplusplus_type_name &rhs) const {
         return cpp_type_type < rhs.cpp_type_type ||
                (cpp_type_type == rhs.cpp_type_type &&
                 name < rhs.name);
    }
};

/** A structure to represent the fully scoped name of a C++ composite
 * type for use in generating function signatures that use that type.
 *
 * This is intended to be a constexpr usable type, but we don't depend
 * on C++11 yet. In C++14, it is possible this will be replaced with
 * introspection/reflection facilities.
 */
struct halideir_handle_cplusplus_type {
    halideir_cplusplus_type_name inner_name;
    std::vector<std::string> namespaces;
    std::vector<halideir_cplusplus_type_name> enclosing_types;

    /// One set of modifiers on a type.
    /// The const/volatile/restrict propertises are "inside" the pointer property.
    enum Modifier : uint8_t {
        Const = 1 << 0,    ///< Bitmask flag for "const"
        Volatile = 1 << 1, ///< Bitmask flag for "volatile"
        Restrict = 1 << 2, ///< Bitmask flag for "restrict"
        Pointer = 1 << 3,  ///< Bitmask flag for a pointer "*"
    };

    /// Qualifiers and indirections on type. 0 is innermost.
    std::vector<uint8_t> cpp_type_modifiers;

    /// References are separate because they only occur at the outermost level.
    /// No modifiers are needed for references as they are not allowed to apply
    /// to the reference itself. (This isn't true for restrict, but that is a C++
    /// extension anyway.) If modifiers are needed, the last entry in the above
    /// array would be the modifers for the reference.
    enum ReferenceType : uint8_t {
        NotReference = 0,
        LValueReference = 1, // "&"
        RValueReference = 2, // "&&"
    };
    ReferenceType reference_type;

    halideir_handle_cplusplus_type(const halideir_cplusplus_type_name &inner_name,
                                 const std::vector<std::string> &namespaces = { },
                                 const std::vector<halideir_cplusplus_type_name> &enclosing_types = { },
                                 const std::vector<uint8_t> &modifiers = { },
                                 ReferenceType reference_type = NotReference)
    : inner_name(inner_name),
      namespaces(namespaces),
      enclosing_types(enclosing_types),
      cpp_type_modifiers(modifiers),
      reference_type(reference_type) {}
};
//@}

template<typename T>
struct halideir_c_type_to_name {
  static const bool known_type = false;
};

#define HALIDEIR_DECLARE_EXTERN_TYPE(TypeType, Type)                      \
    template<> struct halideir_c_type_to_name<Type> {                     \
        static const bool known_type = true;                            \
        static halideir_cplusplus_type_name name() {                      \
            return { halideir_cplusplus_type_name::TypeType, #Type};      \
        }                                                               \
    }

#define HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(T)     HALIDEIR_DECLARE_EXTERN_TYPE(Simple, T)
#define HALIDEIR_DECLARE_EXTERN_STRUCT_TYPE(T)     HALIDEIR_DECLARE_EXTERN_TYPE(Struct, T)
#define HALIDEIR_DECLARE_EXTERN_CLASS_TYPE(T)      HALIDEIR_DECLARE_EXTERN_TYPE(Class, T)
#define HALIDEIR_DECLARE_EXTERN_UNION_TYPE(T)      HALIDEIR_DECLARE_EXTERN_TYPE(Union, T)

HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(bool);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(int8_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(uint8_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(int16_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(uint16_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(int32_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(uint32_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(int64_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(uint64_t);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(float);
HALIDEIR_DECLARE_EXTERN_SIMPLE_TYPE(double);

// You can make arbitrary user-defined types be "Known" using the
// macro above. This is useful for making Param<> arguments for
// Generators type safe. e.g.,
//
//    struct MyFunStruct { ... };
//
//    ...
//
//    HALIDEIR_DECLARE_EXTERN_STRUCT_TYPE(MyFunStruct);
//
//    ...
//
//    class MyGenerator : public Generator<MyGenerator> {
//       Param<const MyFunStruct *> my_struct_ptr;
//       ...
//    };


// Default case (should be only Unknown types, since we specialize for Known types below).
// We require that all unknown types be pointers, and translate them all to void*
// (preserving const-ness and volatile-ness).
template<typename T, bool KnownType>
struct halideir_internal_handle_traits {
    static const halideir_handle_cplusplus_type *type_info(bool is_ptr,
            halideir_handle_cplusplus_type::ReferenceType ref_type) {
        static_assert(!KnownType, "Only unknown types handled here");
        internal_assert(is_ptr) << "Unknown types must be pointers";
        internal_assert(ref_type == halideir_handle_cplusplus_type::NotReference) << "Unknown types must not be references";
        static const halideir_handle_cplusplus_type the_info{
            {halideir_cplusplus_type_name::Simple, "void"},
            {},
            {},
            {
                (uint8_t)(halideir_handle_cplusplus_type::Pointer |
                    (std::is_const<T>::value ? halideir_handle_cplusplus_type::Const : 0) |
                    (std::is_volatile<T>::value ? halideir_handle_cplusplus_type::Volatile : 0))
            },
            halideir_handle_cplusplus_type::NotReference
        };
        return &the_info;
    }
};

// Known types
template<typename T>
struct halideir_internal_handle_traits<T, true> {

    static const halideir_handle_cplusplus_type make_info(bool is_ptr,
                                                        halideir_handle_cplusplus_type::ReferenceType ref_type) {
        halideir_handle_cplusplus_type the_info = {
            halideir_c_type_to_name<typename std::remove_cv<T>::type>::name(),
            {},
            {},
            {
                (uint8_t)((is_ptr ? halideir_handle_cplusplus_type::Pointer : 0) |
                    (std::is_const<T>::value ? halideir_handle_cplusplus_type::Const : 0) |
                    (std::is_volatile<T>::value ? halideir_handle_cplusplus_type::Volatile : 0))
            },
            ref_type
        };
        // Pull off any namespaces
        the_info.inner_name.name =
            Halide::Internal::extract_namespaces(the_info.inner_name.name,
                                                 the_info.namespaces);
        return the_info;
    }

    static const halideir_handle_cplusplus_type *type_info(bool is_ptr,
                                                         halideir_handle_cplusplus_type::ReferenceType ref_type) {
        static const halideir_handle_cplusplus_type the_info = make_info(is_ptr, ref_type);
        return &the_info;
    }
};

/** A type traits template to provide a halideir_handle_cplusplus_type
 * value from a C++ type.
 *
 * Note the type represented is implicitly a pointer.
 *
 * A NULL pointer of type halideir_handle_traits represents "void *".
 * This is chosen for compactness or representation as Type is a very
 * widely used data structure.
 */
template<typename T>
struct halideir_handle_traits {
    // NULL here means "void *". This trait must return a pointer to a
    // global structure. I.e. it should never be freed.
    inline static const halideir_handle_cplusplus_type *type_info() { return nullptr; }
};

template<typename T>
struct halideir_handle_traits<T *> {
    inline static const halideir_handle_cplusplus_type *type_info() {
        return halideir_internal_handle_traits<T, halideir_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(true, halideir_handle_cplusplus_type::NotReference);
     }
};

template<typename T>
struct halideir_handle_traits<T &> {
    inline static const halideir_handle_cplusplus_type *type_info() {
        return halideir_internal_handle_traits<T, halideir_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(false, halideir_handle_cplusplus_type::LValueReference);
    }
};

template<typename T>
struct halideir_handle_traits<T &&> {
    inline static const halideir_handle_cplusplus_type *type_info() {
        return halideir_internal_handle_traits<T, halideir_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(false, halideir_handle_cplusplus_type::RValueReference);
    }
};

template<>
struct halideir_handle_traits<const char *> {
    inline static const halideir_handle_cplusplus_type *type_info() {
        static const halideir_handle_cplusplus_type the_info{
            halideir_cplusplus_type_name(halideir_cplusplus_type_name::Simple, "char"),
            {}, {}, { halideir_handle_cplusplus_type::Pointer |
                      halideir_handle_cplusplus_type::Const}};
        return &the_info;
    }
};

namespace Halide {

struct Expr;

enum QuanMode {
  RND,
  RND_ZERO,
  RND_MIN_INF,
  RND_INF,
  RND_CONV,
  TRN,
  TRN_ZERO};

enum OverMode {
  SAT,
  SAT_ZERO,
  SAT_SYM,
  WRAP,
  WRAP_SM};

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats of various bit-widths (the 'bits' field). They can also
 * be vectors of the same (by setting the 'lanes' field to something
 * larger than one). Front-end code shouldn't use vector
 * types. Instead vectorize a function. */
struct Type {
  private:
    halideir_type_t type;

  public:
    /** Aliases for halideir_type_code_t values for legacy compatibility
     * and to match the Halide internal C++ style. */
    // @{
    static const halideir_type_code_t Int = halideir_type_int;
    static const halideir_type_code_t UInt = halideir_type_uint;
    static const halideir_type_code_t Float = halideir_type_float;
    static const halideir_type_code_t Handle = halideir_type_handle;
    // @}

    QuanMode qmode;
    OverMode omode;

    /** The number of bytes required to store a single scalar value of this type. Ignores vector lanes. */
    int bytes() const {return (bits() + 7) / 8;}

    // Default ctor initializes everything to predictable-but-unlikely values
    Type() : type(Handle, 0, 0, 0), qmode(RND), omode(SAT), handle_type(nullptr) {}


    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    Type(halideir_type_code_t code, int bits, int lanes, int fracs = 0, QuanMode qmode = RND, OverMode omode = SAT,
        const halideir_handle_cplusplus_type *handle_type = nullptr)
        : type(code, (uint16_t)bits, (uint8_t)lanes, (uint8_t)fracs), qmode(RND), omode(SAT), handle_type(handle_type) {
    }

    /** Trivial copy constructor. */
    Type(const Type &that) = default;

    /** Type is a wrapper around halideir_type_t with more methods for use
     * inside the compiler. This simply constructs the wrapper around
     * the runtime value. */
    Type(const halideir_type_t &that, const halideir_handle_cplusplus_type *handle_type = nullptr) 
         : type(that), qmode(RND), omode(SAT), handle_type(handle_type) {}

    /** Unwrap the runtime halideir_type_t for use in runtime calls, etc.
     * Representation is exactly equivalent. */
    operator halideir_type_t() const { return type; }

    /** Return the underlying data type of an element as an enum value. */
    halideir_type_code_t code() const { return (halideir_type_code_t)type.code; }

    /** Return the bit size of a single element of this type. */
    int bits() const { return type.bits; }

    /** Return the fractional bit size of a single element of this type. */
    int fracs() const { return type.fracs; }

    /** Return the number of vector elements in this type. */
    int lanes() const { return type.lanes; }

    /** Return Type with same number of bits and lanes, but new_code for a type code. */
    Type with_code(halideir_type_code_t new_code) const {
        return Type(new_code, bits(), lanes(), fracs(), qmode, omode,
                    (new_code == code()) ? handle_type : nullptr);
    }

    /** Return Type with same type code and lanes, but new_bits for the number of bits. */
    Type with_bits(int new_bits) const {
        return Type(code(), new_bits, lanes(), fracs(), qmode, omode,
                    (new_bits == bits()) ? handle_type : nullptr);
    }

    /** Return Type with same type code and lanes, but new_fracs for the number of frac bits. */
    Type with_fracs(int new_fracs) const {
        return Type(code(), bits(), lanes(), new_fracs, qmode, omode,
                    (new_fracs == fracs()) ? handle_type : nullptr);
    }

    /** Return Type with same type code and number of bits,
     * but new_lanes for the number of vector lanes. */
    Type with_lanes(int new_lanes) const {
        return Type(code(), bits(), new_lanes, fracs(), qmode, omode, handle_type);
    }

    /** Type to be printed when declaring handles of this type. */
    const halideir_handle_cplusplus_type *handle_type;

    /** Is this type boolean (represented as UInt(1))? */
    bool is_bool() const {return code() == UInt && bits() == 1;}

    /** Is this type a vector type? (lanes() != 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_vector() const {return lanes() != 1;}

    /** Is this type a scalar type? (lanes() == 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_scalar() const {return lanes() == 1;}

    /** Is this type a floating point type (float or double). */
    bool is_float() const {return code() == Float;}

    /** Is this type a signed integer type? */
    bool is_int() const {return code() == Int && fracs() == 0;}

    /** Is this type an unsigned integer type? */
    bool is_uint() const {return code() == UInt && fracs() == 0;}

    /** Is this type an signed fixed-point type? */
    bool is_fixed() const {return code() == Int && fracs() >= 0;}

    /** Is this type an unsigned fixed-point type? */
    bool is_ufixed() const {return code() == UInt && fracs() >= 0;}

    /** Is this type an opaque handle type (void *) */
    bool is_handle() const {return code() == Handle;}

    /** Check that the type name of two handles matches. */
    EXPORT bool same_handle_type(const Type &other) const;

    /** Compare two types for equality */
    bool operator==(const Type &other) const {
        return code() == other.code() && bits() == other.bits() && lanes() == other.lanes() &&
            fracs() == other.fracs() && (code() != Handle || same_handle_type(other)) &&
            qmode == other.qmode && omode == other.omode;
    }

    /** Compare two types for inequality */
    bool operator!=(const Type &other) const {
        return code() != other.code() || bits() != other.bits() || lanes() != other.lanes() ||
            fracs() != other.fracs() || (code() == Handle && !same_handle_type(other)) ||
            qmode != other.qmode || omode != other.omode;
    }

    /** Produce the scalar type (that of a single element) of this vector type */
    Type element_of() const {
        return with_lanes(1);
    }

    /** Can this type represent all values of another type? */
    EXPORT bool can_represent(Type other) const;

    /** Can this type represent a particular constant? */
    // @{
    EXPORT bool can_represent(double x) const;
    EXPORT bool can_represent(int64_t x) const;
    EXPORT bool can_represent(uint64_t x) const;
    // @}

    /** Check if an integer constant value is the maximum or minimum
     * representable value for this type. */
    // @{
    EXPORT bool is_max(uint64_t) const;
    EXPORT bool is_max(int64_t) const;
    EXPORT bool is_min(uint64_t) const;
    EXPORT bool is_min(int64_t) const;
    // @}

    /** Return an expression which is the maximum value of this type */
    EXPORT Expr max() const;

    /** Return an expression which is the minimum value of this type */
    EXPORT Expr min() const;
};

/** Constructing a signed integer type */
inline Type Int(int bits, int lanes = 1, int fracs = 0) {
    return Type(Type::Int, bits, lanes, fracs);
}

/** Constructing an unsigned integer type */
inline Type UInt(int bits, int lanes = 1, int fracs = 0) {
    return Type(Type::UInt, bits, lanes, fracs);
}

/** Construct a floating-point type */
inline Type Float(int bits, int lanes = 1) {
    return Type(Type::Float, bits, lanes);
}

/** Construct a boolean type */
inline Type Bool(int lanes = 1) {
    return UInt(1, lanes);
}

/** Construct a handle type */
inline Type Handle(int lanes = 1, const halideir_handle_cplusplus_type *handle_type = nullptr) {
    return Type(Type::Handle, 64, lanes, 0, RND, SAT, handle_type);
}

/** Constructing a signed fixed-point type */
inline Type Fixed(int bits, int fracs, int lanes = 1, QuanMode qmode = RND, OverMode omode = SAT) {
    return Type(Type::Int, bits, lanes, fracs, qmode, omode);
}

/** Constructing a unsigned fixed-point type */
inline Type UFixed(int bits, int fracs, int lanes = 1, QuanMode qmode = RND, OverMode omode = SAT) {
    return Type(Type::UInt, bits, lanes, fracs, qmode, omode);
}

/** Construct the halide equivalent of a C type */
template<typename T>
inline Type type_of() {
    return Type(halideir_type_of<T>(), halideir_handle_traits<T>::type_info());
}

}  // namespace Halide

#endif
