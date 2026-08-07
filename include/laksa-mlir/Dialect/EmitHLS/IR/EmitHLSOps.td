//===- EmitHLSOps.td - EmitHLS dialect ops -----------------*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_OPS
#define EMITHLS_OPS

include "laksa-mlir/Dialect/EmitHLS/EmitHLSEnums.td"
include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.td"
include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.td"
include "laksa-mlir/Dialect/EmitHLS/Interfaces/PragmaInterface.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Structure operations
//===----------------------------------------------------------------------===//

def EmitHLS_IncludeOp : EmitHLS_Op<"include", [
        HasParent<"ModuleOp">]>
{
    let summary = "C/C++ include statement";
    let description = [{
        The `include` operation abstracts the C/C++ include syntax.
        However, this operation does not check the existence of the
        header file.
        This operation is created automatically by a transformation pass
        inside this dialect.
    }];

    let arguments = (ins Arg<StrAttr>:$header);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$header)>
    ];
    
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def EmitHLS_FuncOp : EmitHLS_Op<"func", [
        AutomaticAllocationScope,
        FunctionOpInterface,
        IsolatedFromAbove,
        NoTerminator]>
{
    let summary = "Defines a C/C++ function with a void return type.";
    let description = [{
        The `func` operation defines a C/C++ function with a void return type,
        serving as the node and top-level function unit for HLS code generation.
        The function is identified by its symbol name and is isolated from
        above, meaning it cannot reference values defined outside its scope.
        Function arguments are accessible via the entry block of the body region.
        Example:
        ```
        emithls.func @my_func(%arg0: type, %arg1: type, ...) {
            ops...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                        TypeAttrOf<FunctionType>:$function_type,
                        OptionalAttr<DictArrayAttr>:$arg_attrs,
                        OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region SizedRegion<1>:$body);

    let builders = [
        OpBuilder<(ins "StringRef":$name,
                       "FunctionType":$functionType)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        ::mlir::Region *getCallableRegion();
        ArrayRef<Type> getArgumentTypes();
        ArrayRef<Type> getResultTypes();
    }];
}

def EmitHLS_CallOp : EmitHLS_Op<"call", [
        CallOpInterface,]>
{
    let summary = "Defines a function call.";

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$operands,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Variadic<AnyType>);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands)>
    ];

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallOpInterface Methods
        //===------------------------------------------------------------------===//

        operand_range getArgOperands();
        MutableOperandRange getArgOperandsMutable();
        operand_iterator arg_operand_begin();
        operand_iterator arg_operand_end();
        CallInterfaceCallable getCallableForCallee();
        void setCalleeFromCallable(CallInterfaceCallable callee);
    }];
}

def EmitHLS_VariableOp : EmitHLS_Op<"variable", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Defines a C/C++ style variable.";
    let description = [{
        The `variable` operation defines a C/C++ variable, which is used to
        simplify the translation from `emithls` dialect to HLS code.
        MLIR adopts the Single Static Assignment form, which forbids multiple
        assignments to the same variable.
        To solve this, the `variable` operation and an `update` operation are
        defined, such as:
        ```
        %var = emithls.variable : type
        emithls.update %var, %value : type
        ```
        A `variable` can also be defined as a constant as in C/C++.
    }];

    let arguments = (ins OptionalAttr<AnyAttr>:$init_number,
                         Optional<AnyType>:$init_value,
                         UnitAttr:$is_const);
    let results = (outs AnyType:$variable);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Type":$type,
                       CArg<"Attribute", "{}">:$initNumber,
                       CArg<"bool", "false">:$isConst)>,
        OpBuilder<(ins "Type":$type,
                       CArg<"Value", "{}">:$initValue,
                       CArg<"bool", "false">:$isConst)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        bool hasInit();
    }];
}

def EmitHLS_UpdateOp : EmitHLS_Op<"update", [
        MemoryEffectsOpInterface,
        AttrSizedOperandSegments]>
{
    let summary = "Defines a C/C++ style assignment.";
    let description = [{
        The `update` operation performs a C/C++ style assignment to an
        existing variable.
        This operation also supports element-wise assignment for array
        types using a list of `IndexType` indices.
        ```
        emithls.update %var with %new_value : type <- type
        emithls.update %var with %arr[%indices]
            : type <- !emithls.array<shape x type>
        emithls.update %arr[%indices] with %new_value
            : !emithls.array<shape x type> <- type
        emithls.update %arr0[%indices] with %arr1[%indices]
            : !emithls.array<shape x type> <- !emithls.array<shape x type>
        ```
    }];

    let arguments = (ins AnyType:$variable,
                         Variadic<Index>:$indices,
                         AnyType: $new_value,
                         Variadic<Index>:$new_value_indices);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       "Value":$newValue,
                       CArg<"SmallVector<Value>","{}">:$indices,
                       CArg<"SmallVector<Value>","{}">:$newValueIndices)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

def EmitHLS_ExpressionOp : EmitHLS_Op<"expr", [
        NoRegionArguments,
        SingleBlockImplicitTerminator<"YieldOp">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Represents a combined C/C++ expression.";
    let description = [{
        The `expression` operation defines a set of operations that together
        form a C/C++ style complex expression, such as:
        `int var = (a + b) * c;`.
        This operation returns one value with a scalar type, which is yielded
        from the region using `YieldOp`.
        Nested expressions are also allowed.
        Example:
        ```
        // Assume %lhs, %rhs, %cst are i32 values
        %expr = emithls.expr : i32 {
            %0 = arith.addi %lhs, %cst : i32
            %1 = arith.muli %0, %rhs : i32
            emithls.yield %1 : i32
        }
        ```
    }];

    let results = (outs AnyType:$result);
    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Type":$resultType,
                       "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
    ];

    let assemblyFormat = "attr-dict `:` type($result) $body";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        Operation* getRootOp();
    }];
}

def EmitHLS_YieldOp : EmitHLS_Op<"yield", [
        Pure,
        Terminator,
        HasParent<"ExpressionOp">]>
{
    let summary = "Yields the result value of an expression.";
    let description = [{
        The `yield` operation terminates an expression block with a value,
        which becomes the result of the enclosing expression.
        This operation mimics a combined expression in C/C++.
    }];

    let arguments = (ins AnyType:$value);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$value)>,
        OpBuilder<(ins)>
    ];

    let assemblyFormat = "attr-dict $value `:` type($value)";
}

def EmitHLS_IfOp : EmitHLS_Op<"if", [
        RecursiveMemoryEffects,
        NoRegionArguments,
        NoTerminator]>
{
    let summary = "Defines a C/C style if-statement.";
    let description = [{
        The `if` operation conditionally executes one of two regions based on
        an `i1` condition value, mapping to a C/C++ `if`/`else` statement.
        The `then` region is always executed when the condition is true.
        The optional `else` region is executed when the condition is false;
        if absent, no action is taken on the false path.
        Example:
        ```
        emithls.if %cond {
            ops ...
        } else {
            ops ...
        }
        ```
    }];

    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$thenRegion,
                          MaxSizedRegion<1>:$elseRegion);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$condition,
                       "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
                       CArg<"function_ref<void(OpBuilder &, Location)>",
                            "nullptr">:$elseBuilder)>,
        OpBuilder<(ins "Value":$condition,
                       CArg<"bool", "false">:$ensureElseBlock)>
    ];

    let hasCustomAssemblyFormat = 1;
}

def EmitHLS_ForOp : EmitHLS_Op<"for", [
        RecursiveMemoryEffects,
        NoTerminator,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]>
{
    let summary = "Defines a C/C++ for loop.";
    let description = [{
        The `for` operation defines a counted loop with compile-time constant
        lower bound, upper bound, and step, mapping to a C/C++ `for` loop.
        The loop body receives the current induction variable as a block
        argument and iterates from `lowerBound` to `upperBound` (exclusive)
        in increments of `step`.
        Example:
        ```
        emithls.for %idx0 = 0 to 10 step 1 {
            ops using %idx0 ...
        }
        ```
    }];

    let arguments = (ins IndexAttr:$lowerBound,
                         IndexAttr:$upperBound,
                         IndexAttr:$step);
    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "int64_t":$lowerBound, "int64_t":$upperBound, CArg<"int64_t", "1">:$step,
                       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>","nullptr">:$bodyBuilder)>,
        OpBuilder<(ins "Attribute":$lowerBound, "Attribute":$upperBound, "Attribute":$step,
                       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>","nullptr">:$bodyBuilder)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        Value getInductionVariable();
        int64_t getTripCount();
    }];
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

class EmitHLS_ArithOpBase<string mnemonic, list<Trait> traits = []>
        : EmitHLS_Op< "arith." # mnemonic, traits> {}

class EmitHLS_BinaryArithOp<string mnemonic, list<Trait> traits = []>
        : EmitHLS_ArithOpBase<mnemonic, traits # [Pure, SameOperandsAndResultType]>
{
    let arguments = (ins AnyType:$lhs, AnyType: $rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class EmitHLS_LogicalArithOp<string mnemonic, list<Trait> traits = []>
        : EmitHLS_ArithOpBase<mnemonic, traits # [Pure, SameOperandsAndResultType]>
{
    let arguments = (ins Variadic<I1>:$values);
    let results = (outs I1:$result);
    let assemblyFormat = "$values attr-dict `:` type($result)";
    // Verify if the operands size is correct
    // let hasVerifier = 1;
    let extraClassDeclaration = [{
        LogicalResult verify() {
            if (getValues().size() < 2) {
                return emitOpError("expected at least 2 operands");
            }
            return success();
        }
    }];
}

def EmitHLS_ArithAddOp : EmitHLS_BinaryArithOp<"add", []> {}
def EmitHLS_ArithSubOp : EmitHLS_BinaryArithOp<"sub", []> {}
def EmitHLS_ArithMulOp : EmitHLS_BinaryArithOp<"mul", []> {}
def EmitHLS_ArithDivOp : EmitHLS_BinaryArithOp<"div", []> {}
def EmitHLS_ArithRemOp : EmitHLS_BinaryArithOp<"rem", []> {}
def EmitHLS_ArithAndOp : EmitHLS_BinaryArithOp<"and", []> {}
def EmitHLS_ArithOrOp : EmitHLS_BinaryArithOp<"or", []> {}
def EmitHLS_ArithShlOp : EmitHLS_BinaryArithOp<"shl", []> {}
def EmitHLS_ArithShrOp : EmitHLS_BinaryArithOp<"shr", []> {}
def EmitHLS_ArithMinOp : EmitHLS_BinaryArithOp<"min", []> {}
def EmitHLS_ArithMaxOp : EmitHLS_BinaryArithOp<"max", []> {}

def EmitHLS_ArithLogicalAndOp : EmitHLS_LogicalArithOp<"logical_and", []> {}
def EmitHLS_ArithLogicalOrOp : EmitHLS_LogicalArithOp<"logical_or", []> {}

def EmitHLS_ArithFusedOp : EmitHLS_ArithOpBase<"fused", [
        SameTypeOperands,
        MemoryEffectsOpInterface]>
{
    let summary = "Defines C/C++ style fused assignment.";

    let arguments = (ins EmitHLS_FusedOperatorAttr:$op_code,
                         AnyType:$acc,
                         AnyType:$val);

    let assemblyFormat = "$op_code `,` $acc `,` $val attr-dict `:` type($acc)";
    let hasVerifier = 1;
    
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

def EmitHLS_ArithCastOp : EmitHLS_ArithOpBase<"cast", [
        Pure,
        SameOperandsAndResultShape]>
{
    let summary = "Arithmetic type cast operation.";
    let arguments = (ins AnyType:$from);
    let results = (outs AnyType:$to);
    let assemblyFormat = "$from attr-dict `:` type($from) `to` type($to)";
}

def EmitHLS_ArithCmpOp : EmitHLS_ArithOpBase<"cmp", [
        Pure,
        SameTypeOperands]>
{
    let summary = "Arithmetic comparison with predicate.";
    let arguments = (ins EmitHLS_CmpPredicateAttr:$predicate,
                         AnyType:$lhs,
                         AnyType:$rhs);
    let results = (outs I1:$result);
    let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def EmitHLS_ArithSelectOp : EmitHLS_ArithOpBase<"select",[
        Pure,
        AllTypesMatch<["true_value", "false_value", "result"]>]>
{
    let arguments = (ins I1:$condition,
                         AnyType:$true_value,
                         AnyType:$false_value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($result)";
}

def EmitHLS_ArithDataRangeOp : EmitHLS_ArithOpBase<"data_range", [Pure]>
{
    let arguments = (ins AnyType:$data,
                         Index:$high_bit,
                         Index:$low_bit);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

class EmitHLS_ArrayOpBase<string mnemonic, list<Trait> traits = []>
        : EmitHLS_Op< "array." # mnemonic, traits> {}

def EmitHLS_ArrayReadOp : EmitHLS_ArrayOpBase<"read", [
        TypesMatchWith<"result type must match array type", "array", "result",
                       "::llvm::cast<emithls::ArrayType>($_self).getElementType()">,
        MemoryEffectsOpInterface]>
{
    let summary = "Reads a value from an array";
    let arguments = (ins Arg<EmitHLS_ArrayType>:$array,
                         Variadic<Index>:$indices);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $indices `]` attr-dict `:` qualified(type($array)) `->` type($result)";
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
        
        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        ArrayType getArrayType();
    }];
}

def EmitHLS_ArrayWriteOp : EmitHLS_ArrayOpBase<"write", [
        TypesMatchWith<"result type must match array type", "array", "value",
                       "::llvm::cast<emithls::ArrayType>($_self).getElementType()">,
        MemoryEffectsOpInterface]>
{
    let summary = "Writes a value into an array";
    let arguments = (ins AnyType:$value,
                         Arg<EmitHLS_ArrayType>:$array,
                         Variadic<Index>:$indices);
    let assemblyFormat = "$value `,` $array `[` $indices `]` attr-dict `:` type($value) `->` qualified(type($array))";
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
        
        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        ArrayType getArrayType();
    }];
}

def EmitHLS_ArrayPointerReadOp : EmitHLS_ArrayOpBase<"ptr_read", [
        TypesMatchWith<"result type must match pointer type", "pointer", "result",
                       "::llvm::cast<emithls::PointerType>($_self).getElementType()">,
        MemoryEffectsOpInterface]>
{
    let summary = "Reads a value from an pointer";
    let arguments = (ins Arg<EmitHLS_PointerType>:$pointer,
                         Variadic<Index>:$indices);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$pointer `[` $indices `]` attr-dict `:` qualified(type($pointer)) `->` type($result)";
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

def EmitHLS_ArrayPointerWriteOp : EmitHLS_ArrayOpBase<"ptr_write", [
        TypesMatchWith<"result type must match pointer type", "pointer", "value",
                       "::llvm::cast<emithls::PointerType>($_self).getElementType()">,
        MemoryEffectsOpInterface]>
{
    let summary = "Writes a value into an array";
    let arguments = (ins AnyType:$value,
                         Arg<EmitHLS_PointerType>:$pointer,
                         Variadic<Index>:$indices);
    let assemblyFormat = "$value `,` $pointer `[` $indices `]` attr-dict `:` type($value) `->` qualified(type($pointer))";
    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

//===----------------------------------------------------------------------===//
// Stream operations
//===----------------------------------------------------------------------===//

class EmitHLS_StreamOpBase<string mnemonic, list<Trait> traits = []>
        : EmitHLS_Op< "stream." # mnemonic, traits> {}

def EmitHLS_StreamReadOp : EmitHLS_StreamOpBase<"read", [
        MemoryEffectsOpInterface]>
{
    let summary = "Reads data out from hls::stream.";

    let arguments = (ins AnyTypeOf<[EmitHLS_StreamType, EmitHLS_ArrayType]>:$stream,
                         Variadic<Index>:$indices);
    let results = (outs AnyType:$data);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Type":$resultType,
                       "Value":$stream,
                       CArg<"ValueRange", "">:$indices)>
    ];

    let assemblyFormat = "$stream (`[` $indices^ `]`)? attr-dict `:` qualified(type($stream)) `->` type($data)";
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

def EmitHLS_StreamWriteOp : EmitHLS_StreamOpBase<"write", [
        MemoryEffectsOpInterface]>
{
    let summary = "Writes data into hls::stream.";

    let arguments = (ins AnyType:$data,
                         AnyTypeOf<[EmitHLS_StreamType, EmitHLS_ArrayType]>:$stream,
                         Variadic<Index>:$indices);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$data,
                       "Value":$stream,
                       CArg<"ValueRange", "">:$indices)>
    ];

    let assemblyFormat = "$data `to` $stream (`[` $indices^ `]`)? attr-dict `:` type($data) `->` qualified(type($stream))";
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//
        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

//===----------------------------------------------------------------------===//
// Pragma operations
//===----------------------------------------------------------------------===//

class EmitHLS_PragmaOpBase<string mnemonic, list<Trait> traits = []>
        : EmitHLS_Op< "pragma." # mnemonic, traits # [EmitHLS_PragmaInterface]> {}

def EmitHLS_PragmaArrayPartitionOp : EmitHLS_PragmaOpBase<"array_partition", []>
{
    let summary = "#pragma HLS ARRAY_PARTITION variable= type= factor= dim=";

    let arguments = (ins EmitHLS_ArrayType:$variable,
                         EmitHLS_ArrayPartitionTypeAttr:$part_type,
                         OptionalAttr<I32Attr>:$part_factor,
                         OptionalAttr<I32Attr>:$part_dim);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       "emithls::ArrayPartitionType":$partType,
                       CArg<"std::optional<int32_t>", "{}">:$partFactor,
                       CArg<"std::optional<int32_t>", "{}">:$partDim)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Type getVariableType();
    }];
}

def EmitHLS_PragmaBindStorageOp : EmitHLS_PragmaOpBase<"bind_storage", []>
{
    let summary = "#pragma HLS BIND_STORAGE variable= type= impl=";

    let arguments = (ins AnyTypeOf<[EmitHLS_ArrayType, EmitHLS_StreamType]>:$variable,
                         EmitHLS_BindStorageTypeAttr:$storage_type,
                         EmitHLS_BindStorageImplAttr:$storage_impl);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       "emithls::BindStorageType":$storageType,
                       "emithls::BindStorageImpl":$storageImpl)>
    ];

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Type getVariableType();
    }];
}

def EmitHLS_PragmaDataflowOp : EmitHLS_PragmaOpBase<"dataflow", [NoTerminator]>
{
    let summary = "#pragma HLS DATAFLOW";

    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"function_ref<void(OpBuilder &, Location)>","nullptr">:$bodyBuilder)>
    ];

    let assemblyFormat = "$body attr-dict";
}

def EmitHLS_PragmaPipelineOp : EmitHLS_PragmaOpBase<"pipeline", []>
{
    let summary = "#pragma HLS PIPELINE II=";

    let arguments = (ins I32Attr:$interval,
                         EmitHLS_PipelineStyleAttr:$style);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"int32_t", "1">:$interval,
                       CArg<"emithls::PipelineStyle",
                            "emithls::PipelineStyle::flp">:$style)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def EmitHLS_PragmaInlineOp : EmitHLS_PragmaOpBase<"inline", []>
{
    let summary = "#pragma HLS INLINE off";

    let arguments = (ins OptionalAttr<UnitAttr>:$off);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"bool", "false">:$off)>
    ];
    
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;
}

def EmitHLS_PragmaStreamOp : EmitHLS_PragmaOpBase<"stream", []>
{
    let summary = "#pragma HLS STREAM type=fifo depth=";

    let arguments = (ins AnyTypeOf<[EmitHLS_StreamType, EmitHLS_ArrayType]>:$variable,
                         I32Attr:$depth);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$variable,
                       CArg<"int32_t", "2">:$depth)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Type getVariableType();
    }];
}

def EmitHLS_PragmaTopInterfaceOp : EmitHLS_Op<"top_interface", [EmitHLS_PragmaInterface]>
{
    let summary = "#pragma HLS INTERFACE for the top function";
    let assemblyFormat = "attr-dict";
}

def EmitHLS_PragmaUnrollOp : EmitHLS_PragmaOpBase<"unroll", [HasParent<"ForOp">]>
{
    let summary = "#pragma HLS UNROLL factor=";

    let arguments = (ins OptionalAttr<I32Attr>:$factor);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"std::optional<int32_t>", "">)>
    ];

    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Helper operations
//===----------------------------------------------------------------------===//

class EmitHLS_HelperOpBase<string mnemonic, list<Trait> traits = []>
        : EmitHLS_Op< "helper." # mnemonic, traits> {}

def EmitHLS_MemRefOrArray : AnyTypeOf<[AnyMemRef, EmitHLS_ArrayType],
        "memref or emithls.array type">;

def EmitHLS_HelperLineBufferOp : EmitHLS_HelperOpBase<"linebuf", [
        ParentOneOf<["ForOp", "affine::AffineForOp"]>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Helper to store information of line buffer construction";
    let arguments = (ins EmitHLS_MemRefOrArray:$token_ref,
                         Optional<Index>:$index,
                         I32ArrayAttr:$num_chan,
                         I32Attr:$num_line);
    let results = (outs EmitHLS_MemRefOrArray:$buf_ref);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$tokenRef, "Value":$index, "Type":$linebufType,
                       "ArrayRef<int32_t>":$numChan, "int32_t":$numLine)>
    ];

    let hasCustomAssemblyFormat = 1;
}

def EmitHLS_HelperWindowOp : EmitHLS_HelperOpBase<"window", [
        ParentOneOf<["ForOp", "affine::AffineForOp"]>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Helper to construct a compute windows from line buffer.";
    let arguments = (ins EmitHLS_MemRefOrArray:$buf_ref,
                         Variadic<Index>:$indices);
    let results = (outs EmitHLS_MemRefOrArray:$win_ref);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$linebuf, "ValueRange":$indices,
                       "Type":$windowType)>
    ];

    let hasCustomAssemblyFormat = 1;
}

def EmitHLS_HelperAccumulateOp : EmitHLS_HelperOpBase<"accumulate", [
        ParentOneOf<["ForOp", "affine::AffineForOp"]>]>
{
    let summary = "Helper to mark the accumulate axis in the computation.";
    let arguments = (ins EmitHLS_MemRefOrArray:$accu_ref,
                         EmitHLS_FusedOperatorAttr:$op_code,
                         AnyType:$value,
                         Variadic<Index>:$indices);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$accuRef, "ValueRange":$indices,
                       "FusedOperator":$operatorCode, "Value":$value)>
    ];

    let hasCustomAssemblyFormat = 1;
}

#endif // EMITHLS_OPS
