//===- DFGOps.td - DFG dialect ops --------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_OPS
#define DFG_OPS

include "laksa-mlir/Dialect/DFG/DFGEnums.td"
include "laksa-mlir/Dialect/DFG/IR/DFGBase.td"
include "laksa-mlir/Dialect/DFG/IR/DFGTypes.td"
include "laksa-mlir/Dialect/DFG/Interfaces/EdgeInterface.td"
include "laksa-mlir/Dialect/DFG/Interfaces/GraphInterface.td"
include "laksa-mlir/Dialect/DFG/Interfaces/NodeInterface.td"

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
// DFG node/graph operations.
//===----------------------------------------------------------------------===//

def DFG_ProcessOp : DFG_Op<"process", [
        DFG_NodeInterface,
        AffineScope,
        AutomaticAllocationScope,
        IsolatedFromAbove,
        HasParent<"ModuleOp">,
        NoTerminator,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]>
{
    let summary = "Defines a process in the KPN MoC";
    let description = [{
        The `process` operation defines one node (actor) in the KPN dataflow model.
        A process is a symbol at module scope and can be connected to other processes
        through its input and output ports.

        The port interface is described by `function_type`:
        - function inputs represent incoming channel ends
        - function results represent outgoing channel ends

        A `process` can have one region with other IRs inside or none with a link to a library call.

        Example:
        ```
        dfg.process @node inputs(%in0: !dfg.output<type>, %in1: !dfg.output<type>)
                          outputs(%out0: !dfg.input<type>, %out1: !dfg.input<type>)
        {
            ops ...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type,
                         // the multiplicity of the operation, i.e. how many elements from each channel are required for execution
                         DefaultValuedAttr<DenseI64ArrayAttr, "{}">: $multiplicity
                    );

    // The FunctionOpInterface requires zero or one region attached to the process
    let regions = (region AnyRegion: $body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$functionType,
                       CArg<"ArrayRef<int64_t>", "{}">:$multiplicity,
                       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                            "nullptr">:$bodyBuilder)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // NodeInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName();
        unsigned getNumInputPorts();
        Value getInputPort(unsigned index);
        SmallVector<Type> getInputPortTypes();
        SmallVector<Value> getInputPorts();
        unsigned getNumOutputPorts();
        Value getOutputPort(unsigned index);
        SmallVector<Type> getOutputPortTypes();
        SmallVector<Value> getOutputPorts();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration();

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        bool isExternal();
    }];
}

def DFG_LoopOp : DFG_Op<"loop", [
        AutomaticAllocationScope,
        AttrSizedOperandSegments,
        RecursiveMemoryEffects,
        NoTerminator,
        HasParent<"ProcessOp">]>
{
    let summary = "Defines a list of input/output ports may be closed.";

    let description = [{
        The `loop` operation defines a list of input or output ports
        that will be closed later. The close behavior will terminate
        the process and populate to other processes so that the whole
        dataflow graph will be shut down.

        Example:
        ```
        dfg.process @add inputs(%a: !dfg.output<type>) outputs(%b: !dfg.input<type>)
        {
            dfg.loop inputs(%a: !dfg.output<type>) outputs(%b: !dfg.input<type>) {
                ops ...
            }
        }
        ```
    }];

    let arguments = (ins Variadic<DFG_OutputType>:$input_channels,
                         Variadic<DFG_InputType>:$output_channels);
    let regions = (region MinSizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "ValueRange":$inputChannels,
                       "ValueRange":$outputChannels,
                       "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        FunctionType getFunctionType();
    }];
}

def DFG_OperatorOp : DFG_Op<"operator", [
        DFG_NodeInterface,
        AffineScope,
        AutomaticAllocationScope,
        IsolatedFromAbove,
        HasParent<"ModuleOp">,
        SingleBlockImplicitTerminator<"OutputOp">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]>
{
    let summary = "Defines an DFG node in SDF MoC";

    let description = [{
        The `operator` operation defines one node (actor) in the SDF dataflow model.
        An operator is a special case of `process`, which only write values into output
        ports once at the end.

        Example:
        ```
        dfg.operator @node inputs(%in0: i32, %in1: i32) outputs(%out0: i32, %out1: i32)
        {
            ops ...
            dfg.output ...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type);
    let regions = (region AnyRegion: $body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$functionType,
                       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                            "nullptr">:$bodyBuilder)>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // NodeInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName();
        unsigned getNumInputPorts();
        Value getInputPort(unsigned index);
        SmallVector<Type> getInputPortTypes();
        SmallVector<Value> getInputPorts();
        unsigned getNumOutputPorts();
        Value getOutputPort(unsigned index);
        SmallVector<Type> getOutputPortTypes();
        SmallVector<Value> getOutputPorts();
        FunctionType getBaseFunctionType();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration();

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        bool isExternal();
    }];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def DFG_RegionOp : DFG_Op<"region", [
        DFG_NodeInterface,
        DFG_GraphInterface,
        IsolatedFromAbove,
        NoTerminator,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
    ]>
{
    let summary = "Defines a region(subgraph) in the DFG.";

    let description = [{
        The `region` operation defines one node(subgraph) or the entire graph in the
        dataflow graph model.
        A region can be embedded into another region as subgraph, which content can
        only be an instantiation of a node(processes/operators/regions) and their
        connections(channels).
        A region's input/output ports must be connected/used.

        Example:
        ```
        dfg.region @node inputs(%in0: !dfg.output<type>, %in1: !dfg.output<type>)
                         outputs(%out0: !dfg.input<type>, %out1: !dfg.input<type>)
        {
            %0:2 = dfg.channel ...
            dfg.instantiate @process ...
            dfg.instantiate @operator ...
            dfg.embed @region ...
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr: $sym_name,
                         TypeAttrOf<FunctionType>: $function_type);
    let regions = (region SizedRegion<1>:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$functionType,
                       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                            "nullptr">:$bodyBuilder)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // NodeInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName();
        unsigned getNumInputPorts();
        Value getInputPort(unsigned index);
        SmallVector<Type> getInputPortTypes();
        SmallVector<Value> getInputPorts();
        unsigned getNumOutputPorts();
        Value getOutputPort(unsigned index);
        SmallVector<Type> getOutputPortTypes();
        SmallVector<Value> getOutputPorts();

        //===------------------------------------------------------------------===//
        // GraphInterface Methods
        //===------------------------------------------------------------------===//

        std::string getGraphName();
        bool isSubGraph();
        SmallVector<Operation*> getGraphNodes();
        SmallVector<Operation*> getGraphSubGs();
        SmallVector<Operation*> getGraphEdges();
    }];
}

//===----------------------------------------------------------------------===//
// DFG edge operations.
//===----------------------------------------------------------------------===//

def DFG_ChannelOp : DFG_Op<"channel", [
        DFG_EdgeInterface,
        MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Defines a channel with one input and one output port";

    let description = [{
        The `channel` operation produces a typed channel (i.e. an edge in the
        dataflow graph) that links two nodes (i.e. process or operator) in the
        dataflow graph.
        The input of a channel will be connected to an output port of a node,
        which means its type is dfg::InputType, same for output port.

        A channel can also be sized manually with a 32-bit integer, otherwise,
        it indicates an unbounded FIFO channel.
        The token flows in the channel can also be of memref/tensor type, which
        implicitly enables multiple read/write as in KPN MoC.

        Examples:
        ```
        %input, %output = dfg.channel() : type
        $input, %output = dfg.channel(size) : shape x type
        ```
    }];

    let arguments = (ins TypeAttr:$token_type,
                         OptionalAttr<I32Attr>:$buffer_size);
    let results = (outs DFG_InputType:$input_port,
                        DFG_OutputType:$output_port);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$tokenType,
                       CArg<"std::optional<int32_t>", "std::nullopt">:$bufferSize)>,
        OpBuilder<(ins "Type":$type,
                       CArg<"std::optional<int32_t>", "std::nullopt">:$bufferSize)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // EdgeInterface Methods
        //===------------------------------------------------------------------===//

        Operation* getInputConnectedOp();
        Operation* getOutputConnectedOp();

        //===------------------------------------------------------------------===//
        // MemoryEffectsOpInterface Methods
        //===------------------------------------------------------------------===//

        void getEffects(
                SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                &effects);
    }];
}

//===----------------------------------------------------------------------===//
// DFG instance operations.
//===----------------------------------------------------------------------===//

def DFG_InstantiateOp : DFG_Op<"instantiate", [
        DFG_NodeInterface,
        AttrSizedOperandSegments]>
{
    let summary = "Instantiates an DFG node.";

    let description = [{
        The `instantiate` operation instantiate an operator or process in a
        region with a set of inputs and outputs.
        InstantiateOp creates a DFG node with a set of input and output values.

        ```
        dfg.operator @operator inputs(...) outputs(...) {...}
        dfg.process @process inputs(...) outputs(...) {...}
        dfg.region @graph inputs(...) outputs(...) {
            dfg.instantiate @operator inputs(...) outputs(...) : (...) -> (...)
            dfg.instantiate @process inputs(...) outputs(...) : (...) -> (...)
        }
        ```
    }];

    let arguments = (ins SymbolRefAttr:$actor,
                         Variadic<DFG_OutputType>:$inputs,
                         Variadic<DFG_InputType>:$outputs,
                         OptionalAttr<DFG_OffloadHardwareAttr>:$offloaded);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$actorName, "ValueRange":$inputs, "ValueRange":$outputs,
                       CArg<"std::optional<OffloadHardware>","std::nullopt">:$offloaded)>,
        OpBuilder<(ins "ProcessOp":$actorProcess, "ValueRange":$inputs, "ValueRange":$outputs,
                       CArg<"std::optional<OffloadHardware>","std::nullopt">:$offloaded)>,
        OpBuilder<(ins "OperatorOp":$actorOperator, "ValueRange":$inputs, "ValueRange":$outputs,
                       CArg<"std::optional<OffloadHardware>","std::nullopt">:$offloaded)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // NodeInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName();
        unsigned getNumInputPorts();
        Value getInputPort(unsigned index);
        SmallVector<Type> getInputPortTypes();
        SmallVector<Value> getInputPorts();
        unsigned getNumOutputPorts();
        Value getOutputPort(unsigned index);
        SmallVector<Type> getOutputPortTypes();
        SmallVector<Value> getOutputPorts();

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        Operation* getInstantiatedOperation();
        FunctionType getFunctionType();
    }];
}

def DFG_EmbedOp : DFG_Op<"embed", [
        DFG_NodeInterface,
        DFG_GraphInterface,
        AttrSizedOperandSegments]>
{
    let summary = "Embed a region as a subgraph in another.";

    let description = [{
        The `embed` operation embed a region into another region.
        The syntax and usage of this operation is similar to `instantiate` operation.

        ```
        dfg.region @child inputs(...) outputs(...) {...}
        dfg.region @parent inputs(...) outputs(...) {
            dfg.embed @child inputs(...) outputs(...) : (...) -> (...)
        }
        ```
    }];

    let arguments = (ins SymbolRefAttr:$actor,
                         Variadic<DFG_OutputType>:$inputs,
                         Variadic<DFG_InputType>:$outputs,
                         OptionalAttr<DFG_OffloadHardwareAttr>:$offloaded);
    
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "StringRef":$actorName, "ValueRange":$inputs, "ValueRange":$outputs,
                       CArg<"std::optional<OffloadHardware>","std::nullopt">:$offloaded)>,
        OpBuilder<(ins "RegionOp":$actorRegion, "ValueRange":$inputs, "ValueRange":$outputs,
                       CArg<"std::optional<OffloadHardware>","std::nullopt">:$offloaded)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // NodeInterface Methods
        //===------------------------------------------------------------------===//

        std::string getNodeName();
        unsigned getNumInputPorts();
        Value getInputPort(unsigned index);
        SmallVector<Type> getInputPortTypes();
        SmallVector<Value> getInputPorts();
        unsigned getNumOutputPorts();
        Value getOutputPort(unsigned index);
        SmallVector<Type> getOutputPortTypes();
        SmallVector<Value> getOutputPorts();

        //===------------------------------------------------------------------===//
        // GraphInterface Methods
        //===------------------------------------------------------------------===//

        std::string getGraphName();
        bool isSubGraph();

        //===------------------------------------------------------------------===//
        // Custom Methods
        //===------------------------------------------------------------------===//

        Operation* getEmbeddedOperation();
        FunctionType getFunctionType();
    }];
}

//===----------------------------------------------------------------------===//
// DFG structure operations.
//===----------------------------------------------------------------------===//

def DFG_OutputOp : DFG_Op<"output", [
        Terminator,
        HasParent<"OperatorOp">]>
{
    let summary = "Terminator for an OperatorOp";

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;
}

def DFG_PullOp : DFG_Op<"pull", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Pulls a token from an input port of a node.";

    let description = [{
        The `pull` operation reads a token out from an input port of a node,
        which is connected via a channel to an output port of another node.
        An optional set of indices must be used if the read port has a shape,
        indicating a set of ports to be chosen from.

        Example:
        ```
        dfg.process @pull inputs(%in0: !dfg.output<type>, %in1: !dfg.output<shapextype>)
        {
            %0 = arith.constant ... : index
            %token0 = dfg.pull %in0 : !dfg.output<type>
            %token1 = dfg.pull %in1[%0] : !dfg.output<shapextype>
        }
        ```
    }];

    let arguments = (ins DFG_OutputType:$read_port,
                         Variadic<Index>:$indices);
    let results = (outs AnyType:$token);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$readPort,
                       CArg<"ValueRange","{}">:$indices)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def DFG_PullAsTensorOp : DFG_Op<"pull_as_tensor", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Pulls as a tensor from a shaped input port of a node.";

    let arguments = (ins DFG_OutputType:$read_port);
    let results = (outs AnyRankedTensor:$token_tensor);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$readPort)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def DFG_PullAsMemRefOp : DFG_Op<"pull_as_memref", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>
{
    let summary = "Pulls as a memref from a shaped input port of a node.";

    let arguments = (ins DFG_OutputType:$read_port);
    let results = (outs AnyMemRef:$token_memref);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$readPort)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def DFG_PushOp : DFG_Op<"push", []>
{
    let summary = "Pushes a token to an output port of a node.";

    let description = [{
        The `push` operation writes a token to an output port of a node,
        which is connected via a channel to an input port of another node.
        An optional set of indices must be used if the port has a shape.

        Example:
        ```
        dfg.process @pull outputs(%out0: !dfg.input<type>, %out1: !dfg.input<shapextype>)
        {
            %0 = arith.constant ... : index
            %1 = arith.constant ... : type
            %token0 = dfg.push %1 to %out0 : !dfg.input<type>
            %token1 = dfg.pull %1 to %out1[%0] : !dfg.input<shapextype>
        }
        ```
    }];

    let arguments = (ins AnyType:$token,
                         DFG_InputType:$write_port,
                         Variadic<Index>:$indices);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$writePort,
                       "Value":$readPort,
                       CArg<"ValueRange","{}">:$indices)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def DFG_PushTensorOp : DFG_Op<"push_tensor", []>
{
    let summary = "Pushes a tensor to a shaped output port of a node.";

    let arguments = (ins AnyRankedTensor:$token_tensor,
                         DFG_InputType:$write_port);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$tokenTensor,"Value":$writePort)>
    ];

    let hasCustomAssemblyFormat = 1;
}

def DFG_PushMemRefOp : DFG_Op<"push_memref", []>
{
    let summary = "Pushes a memref to a shaped output port of a node.";

    let arguments = (ins AnyMemRef:$token_memref,
                         DFG_InputType:$write_port);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$tokenMemRef,"Value":$writePort)>
    ];

    let hasCustomAssemblyFormat = 1;
}

#endif // DFG_OPS
