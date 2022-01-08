LEVEL = ../../..
include ../../../Makefile.config 

TBLGEN = $(HCL_PKG_LLVM_BIN)/mlir-tblgen
MLIR_INC = $(HCL_PKG_LLVM_PRJ)/mlir/include

OPS = $(wildcard */*Ops.td)

OPS_DIALECT_H_INCS = $(patsubst %Ops.td, %OpsDialect.h.inc, $(OPS))
OPS_TYPES_H_INCS = $(patsubst %Ops.td, %OpsTypes.h.inc, $(OPS))
OPS_TYPES_CPP_INCS = $(patsubst %Ops.td, %OpsTypes.cpp.inc, $(OPS))
OPS_H_INCS = $(patsubst %Ops.td, %Ops.h.inc, $(OPS))
OPS_CPP_INCS = $(patsubst %Ops.td, %Ops.cpp.inc, $(OPS))

INCS = $(OPS_DIALECT_H_INCS) $(OPS_TYPES_H_INCS) $(OPS_TYPES_CPP_INCS) $(OPS_H_INCS) $(OPS_CPP_INCS)

all: $(INCS)

%Ops.h.inc: %Ops.td
	$(TBLGEN) -gen-op-decls -o=$@ $< -I$(MLIR_INC)

%Ops.cpp.inc: %Ops.td
	$(TBLGEN) -gen-op-defs -o=$@ $< -I$(MLIR_INC)

%OpsTypes.h.inc: %Ops.td
	$(TBLGEN) -gen-typedef-decls -o=$@ $< -I$(MLIR_INC)

%OpsTypes.cpp.inc: %Ops.td
	$(TBLGEN) -gen-typedef-defs -o=$@ $< -I$(MLIR_INC)

%OpsDialect.h.inc: %Ops.td
	$(TBLGEN) -gen-dialect-decls -o=$@ $< -I$(MLIR_INC)

clean:
	rm $(INCS)
