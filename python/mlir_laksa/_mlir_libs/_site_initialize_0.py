def context_init_hook(context):
    from ._mlirDialectsDFG import dfg as _dfg
    from ._mlirDialectsEmitHLS import emithls as _emithls

    _dfg.register_dialect(context, load=True)
    _dfg.register_bufferizable_op_interface_external_models(context)
    _emithls.register_dialect(context, load=True)
