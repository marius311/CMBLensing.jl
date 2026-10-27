using CMBLensing, Adapt, LinearAlgebra, Test, Zygote
using CMBLensing.StaticArrays: SVector

# A nonstandard wrapper with a dense parent must retain the generic fallback.
struct FieldTupleAdaptUnknown{A<:AbstractVector{Float32}} <: AbstractVector{Float32}
    data::A
end
Base.size(x::FieldTupleAdaptUnknown) = size(x.data)
Base.getindex(x::FieldTupleAdaptUnknown,i::Int) = x.data[i]
Base.parent(x::FieldTupleAdaptUnknown) = x.data
Adapt.adapt_structure(to,x::FieldTupleAdaptUnknown) = FieldTupleAdaptUnknown(adapt(to,x.data))

ft_adapt_test_backend(arr) = CMBLensing.basetype(typeof(parent(arr)))

function ft_adapt_test_check(original::CMBLensing.BaseField, expected::CMBLensing.BaseField, got::CMBLensing.BaseField)
    @test ft_adapt_test_backend(got.arr) === ft_adapt_test_backend(original.arr)
    @test ft_adapt_test_backend(got.metadata.ℓx) === ft_adapt_test_backend(original.arr)
    @test eltype(got.arr) === eltype(expected.arr)
    @test Array(got.arr) ≈ Array(expected.arr)
end
function ft_adapt_test_check(original::AbstractArray, expected::AbstractArray, got::AbstractArray)
    @test ft_adapt_test_backend(got) === ft_adapt_test_backend(original)
    @test eltype(got) === eltype(expected)
    @test Array(got) ≈ Array(expected)
end
function ft_adapt_test_check(original::Number, expected::Number, got::Number)
    @test typeof(got) === typeof(expected)
    @test got == expected
end
function ft_adapt_test_check(original::FieldTuple, expected::FieldTuple, got::FieldTuple)
    @test keys(getfield(got,:fs)) == keys(getfield(original,:fs))
    map(ft_adapt_test_check, getfield(original,:fs), getfield(expected,:fs), getfield(got,:fs))
end

ft_adapt_test_tuple(c) = FieldTuple(; map=c[1], fourier=c[2], vector=c[3], scalar=c[4],
    nested=FieldTuple(; map=c[5], scalar=c[6]))
function ft_adapt_test_scaled(a,c)
    m,f,v,s,n,ns=c
    FieldTuple(; map=a*m, fourier=a*f, vector=a.*v, scalar=a*s,
        nested=FieldTuple(; map=a*n, scalar=a*ns))
end
ft_adapt_test_objective(a,c,target) = real(sum(diag(adapt(target,Diagonal(ft_adapt_test_scaled(a,c))))))
function ft_adapt_test_partial_objective(a,c,target)
    m,f,v,s = c[1:4]
    components = FieldTuple(;map=a*m, fourier=a*f, vector=a.*v, scalar=a*s)
    sum(diag(adapt(target,Diagonal(components))).fs.map)
end

@testset "FieldTuple diagonal adaptation pullback" begin
    mapcpu=FlatMap(reshape(Float32.(1:16)./10,4,4))
    fouriercpu=Fourier(FlatMap(reshape(Float64.(1:16)./12,4,4)))
    vectorcpu=view([0.0,0.25,-1.0,2.5,0.0],2:4)
    nestedcpu=FlatMap(reshape(Float32.(16:-1:1)./20,4,4))
    viewcpu=CMBLensing.BaseField{Map}(view(mapcpu.arr,:,:),mapcpu.metadata)
    cpu=(viewcpu,fouriercpu,vectorcpu,0.7,nestedcpu,0.2)
    analytic=real(sum(mapcpu)+sum(fouriercpu)+sum(vectorcpu)+0.7+sum(nestedcpu)+0.2)
    if storage == Array
        cases=((cpu,Array),)
    else
        gpu_map=adapt(storage,viewcpu)
        gpu_view=CMBLensing.BaseField{Map}(view(gpu_map.arr,:,:),gpu_map.metadata)
        gpu_vector=adapt(storage,vectorcpu)
        gpu=(gpu_view,adapt(storage,fouriercpu),view(gpu_vector,:),
             0.7,adapt(storage,nestedcpu),0.2)
        mixed=(viewcpu,gpu[2],vectorcpu,0.7,gpu[5],0.2)
        cases=((cpu,Array),(cpu,storage),(gpu,Array),(mixed,storage),(mixed,Array))
    end
    seed=FieldTuple(; map=FlatMap(ones(Float64,4,4)),
        fourier=Fourier(FlatMap(ones(Float32,4,4))),vector=Float32[2,3,4],scalar=Float32(0.4),
        nested=FieldTuple(; map=FlatMap(ones(Float64,4,4)),scalar=Float32(0.9)))
    for (components,target) in cases
        input=Diagonal(ft_adapt_test_tuple(components))
        _,back=Zygote.pullback(x->adapt(target,x),input)
        target_seed=adapt(target,Diagonal(seed))
        returned=only(back(target_seed))
        ft_adapt_test_check(diag(input),diag(target_seed),diag(returned))
        for zero in (nothing,CMBLensing.ChainRulesCore.ZeroTangent())
            z=back(zero)
            @test isnothing(z) || (length(z)==1 &&
                (isnothing(z[1]) || z[1] isa CMBLensing.ChainRulesCore.AbstractZero))
        end
        a=1.25; h=1e-5
        reverse_grad=gradient(a->ft_adapt_test_objective(a,components,target),a)[1]
        finite=(ft_adapt_test_objective(a+h,components,target)-ft_adapt_test_objective(a-h,components,target))/(2h)
        @test !iszero(analytic)
        @test isapprox(reverse_grad,analytic;rtol=5e-5,atol=5e-5)
        @test isapprox(reverse_grad,finite;rtol=5e-5,atol=5e-5)

        # Only the map component contributes; the other components must
        # propagate structural zeros without changing the scalar gradient.
        partial=gradient(a->ft_adapt_test_partial_objective(a,components,target),a)[1]
        partial_finite=(ft_adapt_test_partial_objective(a+h,components,target)-
                        ft_adapt_test_partial_objective(a-h,components,target))/(2h)
        @test isapprox(partial,sum(mapcpu);rtol=5e-5,atol=5e-5)
        @test isapprox(partial,partial_finite;rtol=5e-5,atol=5e-5)
    end

    # Ranges, static arrays and unknown wrappers retain the generic fallback;
    # recognized nested array views use their original dense storage backend.
    base=Float32.(1:12)
    nested=reshape(view(reshape(view(base,2:9),2,4),:,2:3),:)
    @test !(parent(nested) isa DenseArray)
    wrappers=Diagonal(FieldTuple(; range=1f0:1f0:4f0,
        fixed=SVector{4,Float64}(1,2,3,4), nested,
        unknown=FieldTupleAdaptUnknown(Float32[1,2,3,4])))
    wrapper_seed=Diagonal(FieldTuple(; range=Float64[4,3,2,1],
        fixed=Float32[1,3,5,7], nested=Float64[2,4,6,8],unknown=Float64[8,6,4,2]))
    for target in (storage == Array ? (Array,) : (Array,storage))
        _,back=Zygote.pullback(x->adapt(target,x),wrappers)
        target_seed=adapt(target,wrapper_seed)
        returned=diag(only(back(target_seed))).fs
        expected=diag(target_seed).fs
        for name in (:range,:fixed,:unknown)
            @test typeof(getproperty(returned,name)) === typeof(adapt(Diagonal,getproperty(expected,name)))
            @test Array(getproperty(returned,name)) == Array(getproperty(expected,name))
        end
        @test ft_adapt_test_backend(returned.nested) === Array
        @test eltype(returned.nested) === eltype(expected.nested)
        @test Array(returned.nested) == Array(expected.nested)
    end

    cpu_buffer=vcat(vec(mapcpu.arr),vec(mapcpu.arr))
    cpu_nested=reshape(view(reshape(view(cpu_buffer,1:16),2,8),:,1:8),4,4)
    cpu_nested_field=CMBLensing.BaseField{Map}(cpu_nested,mapcpu.metadata)
    if storage == Array
        nested_field_cases=((cpu_nested_field,Array,Array),)
    else
        gpu_map=adapt(storage,mapcpu)
        gpu_flat=vec(gpu_map.arr)
        gpu_buffer=reshape(vcat(gpu_flat,gpu_flat,gpu_flat,gpu_flat),8,8)
        gpu_nested=PermutedDimsArray(view(gpu_buffer,1:2:8,1:2:8),(2,1))
        @test !(parent(gpu_nested) isa DenseArray)
        gpu_nested_field=CMBLensing.BaseField{Map}(gpu_nested,gpu_map.metadata)
        nested_field_cases=((cpu_nested_field,storage,Array),(gpu_nested_field,Array,storage))
    end
    for (field,target,backend) in nested_field_cases
        input=Diagonal(FieldTuple(;map=field,scalar=0.7))
        seed=adapt(target,Diagonal(FieldTuple(;map=FlatMap(ones(Float64,4,4)),scalar=Float32(0.4))))
        _,back=Zygote.pullback(x->adapt(target,x),input)
        returned=diag(only(back(seed))).fs
        @test ft_adapt_test_backend(returned.map.arr) === backend
        @test ft_adapt_test_backend(returned.map.metadata.ℓx) === backend
        @test eltype(returned.map.arr) === Float64
        @test Array(returned.map.arr) == Array(diag(seed).fs.map.arr)
        @test returned.scalar === diag(seed).fs.scalar
    end

    # Dense seeds must keep the generic rule's representation and values.
    input=Diagonal(ft_adapt_test_tuple(cpu))
    _,back=Zygote.pullback(x->adapt(Array,x),input)
    dense=Matrix{Float64}(I,length(diag(input)),length(diag(input)))
    dense[1,2]=0.5
    returned=only(back(dense))
    @test returned isa Matrix{Float64}
    @test returned == dense

    # Positional tuples and unsupported implicit fields retain their structure.
    positional=Diagonal(FieldTuple((mapcpu,fouriercpu)))
    _,positional_back=Zygote.pullback(x->adapt(Array,x),positional)
    pos_result=only(positional_back(positional))
    @test diag(pos_result).fs isa Tuple
    @test diag(pos_result).fs[1].arr == mapcpu.arr
    @test diag(pos_result).fs[2].arr == fouriercpu.arr
    implicit=diag(BandPassOp(0:3,ones(4)))
    implicit_input=Diagonal(FieldTuple(; map=mapcpu, derivative=implicit))
    _,implicit_back=Zygote.pullback(x->adapt(Array,x),implicit_input)
    implicit_result=only(implicit_back(implicit_input))
    @test diag(implicit_result).derivative === implicit
    @test diag(implicit_result).map.arr == mapcpu.arr

    # Differently keyed seeds preserve the generic fallback instead of being
    # silently mapped to unrelated component names.
    mismatched=Diagonal(FieldTuple(; image=mapcpu, harmonic=fouriercpu))
    mismatched_result=only(positional_back(mismatched))
    @test keys(diag(mismatched_result).fs) == (:image,:harmonic)
end
