/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_OPERATION_H_
#define TVM_OPERATION_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "./arithmetic.h"
#include "./buffer.h"
#include "./expr.h"
#include "./ir_operator.h"
#include "./schedule.h"
#include "./tensor.h"

namespace TVM {

using arith::IntSet;

/*!
 * \brief Temporary data structure to store union
 *  of bounds of each axis of Tensor.
 */
struct TensorDom {
  // constructor
  explicit TensorDom(int ndim) : data(ndim) {}
  /*! \brief The domain data */
  std::vector<std::vector<IntSet> > data;
};

/*!
 * \brief Base class of all operation nodes
 */
class OperationNode : public FunctionBaseNode {
 public:
  /*! \brief optional name of the operation */
  std::string name;
  /*! \brief optional tag of the operation */
  std::string tag;
  /*! \return name of the operation */
  const std::string& func_name() const final { return name; }
  /*!
   * \return The list of iteration variable at root
   * \note root_iter_vars dedides the shape of the outputs.
   */
  virtual Array<IterVar> root_iter_vars() const = 0;
  /*!
   * \brief Get data type. i-th output tensor.
   * \param i The output index.
   * \return type of i-th output.
   */
  virtual Type output_dtype(size_t i) const = 0;
  /*!
   * \brief Get shape of i-th output tensor.
   * \param i The output index.
   * \return shape of i-th output.
   */
  virtual Array<Expr> output_shape(size_t i) const = 0;
  /*!
   * \brief List all the input Tensors.
   * \return List of input tensors.
   */
  virtual Array<Tensor> InputTensors() const = 0;
  /*!
   * \brief Replace the input of the operation by pattern specified by rmap.
   *
   * \param self The reference to self.
   * \param rmap The replacement map.
   * \return self if nothing is replaced, otherwise return replaced op.
   */
  virtual Operation ReplaceInputs(
      const Operation& self,
      const std::unordered_map<Tensor, Tensor>& rmap) const = 0;
  /*!
   * \brief Propagate the bounds to inputs
   * \param self The reference to self.
   * \param dom_map the domain map of Variables(corresponds to root_iter_vars)
   * \param out_dom_map The output domain.
   *  The function is only asked to fill the bounds for Tensors that
   *  is already in the out_dom_map
   */
  virtual void PropBoundToInputs(
      const Operation& self,
      const std::unordered_map<const Variable*, IntSet>& dom_map,
      std::unordered_map<Tensor, TensorDom>* out_dom_map) const = 0;
  /*!
   * \brief Gather the bound from output tensor.
   *  Set the range of each root_iter_vars in the op to out_dom_map
   *
   * \param self The reference to self.
   * \param tensor_dom Domain map of Tensor->access set of each dimension.
   * \param out_dom_map The output domain map of each IterVar to be setted.
   */
  virtual void GatherBound(
      const Operation& self,
      const std::unordered_map<Tensor, TensorDom>& tensor_dom,
      std::unordered_map<IterVar, Range>* out_dom_map) const = 0;
  /*!
   * \brief Build the Realize statement that realizes
   *   the op's output tensors.
   * \param stage the op's stage.
   * \param realize_map The realization domain map of the operators.
   * \param body The body that is going to get
   * \return A realization statement that wraps body.
   */
  virtual Stmt BuildRealize(
      const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
      const Stmt& body) const = 0;
  /*!
   * \brief Build the statement that provide the output tensors.
   * \param stage The schedule stage of the op.
   * \param dom_map The domain map of all iteration domains.
   * \param del_trivial_loop Whether eliminate trivial loop with extent of 1
   * \return A statement that add production and wraps consumer.
   */
  virtual Stmt BuildProvide(const Stage& stage,
                            const std::unordered_map<IterVar, Range>& dom_map,
                            bool del_trivial_loop) const = 0;

  static constexpr const char* _type_key = "Operation";

  TVM_DECLARE_BASE_NODE_INFO(OperationNode, Node);
};

/*!
 * \brief A placeholder op represents an input placeholder.
 */
class PlaceholderOpNode : public OperationNode {
 public:
  /*! \brief The shape of the input */
  Array<Expr> shape;
  /*! \brief The data type of the input. */
  Type dtype;
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(
      const Operation& self,
      const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(
      const Operation& self,
      const std::unordered_map<const Variable*, IntSet>& dom_map,
      std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self,
                   const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map) const final;
  Stmt BuildRealize(const Stage& stage,
                    const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage,
                    const std::unordered_map<IterVar, Range>& dom_map,
                    bool del_trivial_loop) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
  }
  static Operation make(std::string name, Array<Expr> shape, Type dtype);

  static constexpr const char* _type_key = "PlaceholderOp";
  TVM_DECLARE_NODE_TYPE_INFO(PlaceholderOpNode, OperationNode);
};

/*!
 * \brief External computation that cannot be splitted.
 */
class ExternOpNode : public OperationNode {
 public:
  /*! \brief The input tensors */
  Array<Tensor> inputs;
  /*! \brief Symbolic placeholder representationinputs */
  Array<Buffer> input_placeholders;
  /*! \brief Symbolic placeholder representation of outputs */
  Array<Buffer> output_placeholders;
  /*! \brief the statement that generates the computation. */
  Stmt body;
  /*! \brief the values that can be used to initialize output tensor. */
  Array<Expr> init_values;
  /*! \brief whether the output tensor is const. */
  bool is_const;

  Array<IterVar> axis;

  /*! \brief constructor */
  ExternOpNode() {}
  // override functions
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  Type output_dtype(size_t i) const final;
  Array<Expr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(
      const Operation& self,
      const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(
      const Operation& self,
      const std::unordered_map<const Variable*, IntSet>& dom_map,
      std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self,
                   const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map) const final;
  Stmt BuildRealize(const Stage& stage,
                    const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage,
                    const std::unordered_map<IterVar, Range>& dom_map,
                    bool del_trivial_loop) const final;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("axis", &axis);
    v->Visit("inputs", &inputs);
    v->Visit("body", &body);
    v->Visit("init_values", &init_values);
    v->Visit("is_const", &is_const);
  }
  EXPORT static Operation make(std::string name, std::string tag,
                               Array<IterVar> axis, Array<Tensor> inputs,
                               Array<Buffer> input_placeholders,
                               Array<Buffer> output_placeholders, Stmt body,
                               Array<Expr> init_values = Array<Expr>(),
                               bool is_const = false);

  static constexpr const char* _type_key = "ExternOp";
  TVM_DECLARE_NODE_TYPE_INFO(ExternOpNode, OperationNode);
};

/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<Expr(const Array<Var>& i)>;

/*! \brief The compute function to specify the inputs source of Tensors */
using FBatchCompute = std::function<Array<Expr>(const Array<Var>& i)>;

/*!
 * \brief create a place holder tensor.
 * \param shape The shape of the tensor.
 * \param dtype the data type of the tensor.
 * \param name The name of the Tensor.
 */
TVM_DLL Tensor placeholder(Array<Expr> shape, Type dtype = Float(32),
                           std::string name = "placeholder");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 */
TVM_DLL Tensor compute(Array<Expr> shape, FCompute fcompute,
                       std::string name = "tensor", std::string tag = "");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensors.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 */
TVM_DLL Array<Tensor> compute(Array<Expr> shape, FBatchCompute fcompute,
                              std::string name = "tensor",
                              std::string tag = "");

// same as compute, specialized for different fcompute function
inline Tensor compute(Array<Expr> shape, std::function<Expr(Var)> f,
                      std::string name = "tensor", std::string tag = "") {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0]); };
  return compute(shape, fc, name, tag);
}
inline Tensor compute(Array<Expr> shape, std::function<Expr(Var, Var)> f,
                      std::string name = "tensor", std::string tag = "") {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1]); };
  return compute(shape, fc, name, tag);
}
inline Tensor compute(Array<Expr> shape, std::function<Expr(Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "") {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1], i[2]); };
  return compute(shape, fc, name, tag);
}
inline Tensor compute(Array<Expr> shape,
                      std::function<Expr(Var, Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "") {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1], i[2], i[3]); };
  return compute(shape, fc, name, tag);
}

// inline function.
inline const OperationNode* Operation::operator->() const {
  return static_cast<const OperationNode*>(node_.get());
}
}  // namespace TVM
#endif  // TVM_OPERATION_H_
