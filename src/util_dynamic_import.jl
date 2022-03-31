
"""
    @dynamic import Foo[: bar, baz, ...]

Allows you to put an import inside of a function. The package won't be
loaded until the first time the function gets run. Note that, in order
to avoid world-age errors, the function is effectively "restarted from
the top" after the first time a given `@dynamic` is encountered, so be
careful not to modify a global state or do costly work inside the
function before the import. The function containing the import will no
longer be inferrable.
"""
macro dynamic(import_statements)

    @assert Base.is_expr(import_statements, :import)
    
    function is_module_loaded_in_world(import_statement)
        imported_module = Base.is_expr(import_statement, :(:)) ? import_statement.args[1].args[1] : import_statement.args[1]
        :(isdefined($__module__, $(QuoteNode(imported_module))) && (try $imported_module.eval(true); catch ex; false; end))
    end
    
    quote
        # if any of the packages are not yet imported in the current
        # world (using the modules' eval method as a check)
        if !$(Expr(:&&, map(is_module_loaded_in_world, import_statements.args)...))

            locals = Base.@locals()

            # do the imports
            @eval $import_statements

            # figure out what method we're in. in the case of a
            # closure, need to reconstruct the callable object
            # including closed over variables
            def = last(Profile.lookup(first(backtrace()))).linfo.def
            func_type = def.sig.parameters[1]
            if hasproperty(func_type, :instance)
                func = func_type.instance
            else
                unwrapped_func_type = Base.unwrap_unionall(func_type)
                closed_over_vars = [
                    T == Core.Box ? Core.Box(locals[p]) : locals[p] 
                    for (p,T) in zip(
                        fieldnames(unwrapped_func_type), 
                        Base.datatype_fieldtypes(unwrapped_func_type
                    ))
                ]
                closed_over_var_types = [
                    typeof(locals[p.name]) 
                    for p in Base.unwrap_unionall(unwrapped_func_type).parameters
                ]
                func = eval(Expr(:new, func_type{closed_over_var_types...}, closed_over_vars...))
            end

            # figure out the original arguments of the current method-call
            args = []
            for (name, T) in Base.arg_decl_parts(def)[2][2:end]
                if name == ""
                    push!(args, eval(Meta.parse(T)).instance)
                elseif endswith(name, "...")
                    append!(args, locals[Symbol(name[1:end-3])])
                else
                    push!(args, locals[Symbol(name)])
                end
            end

            # then call back into the same method with invokelatest so
            # were in a new enough world that we see the imports
            return Base.invokelatest(func, args...)
        end
    end

end
