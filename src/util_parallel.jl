
_mpi_rank() = nothing

@init @require MPIClusterManagers="e7922434-ae4b-11e9-05c5-9780451d2c66" begin

    using .MPIClusterManagers: MPI, start_main_loop, TCP_TRANSPORT_ALL, MPI_TRANSPORT_ALL

    """
    init_MPI_workers()

    Initialize MPI processes as Julia workers. Should be called from all MPI
    processes, and will only return on the master process. 

    `transport` should be `"MPI"` or `"TCP"`, which is by default read from the
    environment variable `JULIA_MPI_TRANSPORT`, and otherwise defaults to `"TCP"`.

    If CUDA is loaded and functional in the Main module, additionally calls
    [`assign_GPU_workers()`](@ref)
    """
    function init_MPI_workers(;
        stdout_to_master = false, 
        stderr_to_master = false,
        transport = get(ENV,"JULIA_MPI_TRANSPORT","TCP"),
        print_info = true,
    )
        
        MPI.Initialized() || MPI.Init()
        size = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)

        if size > 1 && nworkers() == 1

            # workers don't return from this call:
            global _mpi_manager = start_main_loop(
                Dict("TCP"=>TCP_TRANSPORT_ALL,"MPI"=>MPI_TRANSPORT_ALL)[transport],
                stdout_to_master=stdout_to_master,
                stderr_to_master=stderr_to_master
            )
            
            if @isdefined(CUDA) && CUDA.functional()
                assign_GPU_workers(print_info=false)
            end

            print_info && proc_info()

            _mpi_manager

        end

    end

    stop_MPI_workers() = @isdefined(_mpi_manager) && MPIClusterManagers.stop_main_loop(_mpi_manager)

    _mpi_rank() = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : nothing
    
end


"""
    assign_GPU_workers(;print_info=true, use_master=false, remove_oversubscribed_workers=false)

Assign each Julia worker process a unique GPU using `CUDA.device!`.
Works with workers which may be distributed across different hosts,
and each host can have multiple GPUs.

If a unique GPU cannot be assigned, that worker is removed if
`remove_oversubscribed_workers` is true, otherwise an error is thrown.

`use_master` controls whether the master process counts as having been
assigned a GPU (if false, one of the workers may be assigned the same
GPU as the master)
"""
function assign_GPU_workers(;print_info=true, use_master=false, remove_oversubscribed_workers=false)
    if nprocs() > 1
        @everywhere @eval Main using Distributed, CUDA, CMBLensing
        master_uuid = @eval Main CUDA.uuid(device())
        accessible_gpus = Dict(asyncmap(workers()) do id
            @eval Main @fetchfrom $id begin
                ds = CUDA.devices()
                # put master's GPU last so we don't double up on it unless we need to
                $id => sort((CUDA.deviceid.(ds) .=> CUDA.uuid.(ds)), by=(((k,v),)->v==$master_uuid ? Inf : k))
            end
        end)
        claimed = use_master ? Set([master_uuid]) : Set()
        assignments = Dict(map(workers()) do myid
            for (gpu_id, gpu_uuid) in accessible_gpus[myid]
                if !(gpu_uuid in claimed)
                    push!(claimed, gpu_uuid)
                    return myid => gpu_id
                end
            end
            if remove_oversubscribed_workers
                rmprocs(myid)
                return myid => nothing
            else
                error("Can't assign a unique GPU to every worker, process $myid has no free GPUs left.")
            end
        end)
        @everywhere workers() device!($assignments[myid()])
    end
    print_info && proc_info()
end


"""
    proc_info()

Returns string showing info about available processes.
"""
function proc_info()
    @eval Main using Distributed
    lines = @eval Main map(procs()) do id
        @fetchfrom id begin
            info = ["myid = $id"]
            !isnothing(CMBLensing._mpi_rank()) && push!(info, "mpi-rank = $(CMBLensing._mpi_rank())")
            push!(info, "host = $(gethostname())")
            @isdefined(CUDA) && push!(info, "device = $(sprint(io->show(io, MIME("text/plain"), CUDA.device()))) $(split(string(CUDA.uuid(CUDA.device())),'-')[1]))")
            " ("*join(info, ", ")*")"
        end
    end
    @info join(["Processes ($(nprocs())):"; lines], "\n")
end


# a ProgressMeter that can be advanced from any remote worker

struct DistributedProgress <: ProgressMeter.AbstractProgress
    channel :: RemoteChannel{Channel{Any}}
end

function DistributedProgress(args...; kwargs...)
    pbar = Progress(args...; kwargs...)
    channel = RemoteChannel(()->Channel(), 1)
    @async begin
        while (x = take!(channel)) != nothing
            func, args, kwargs = x
            func(pbar, args...; kwargs...)
        end
        finish!(pbar)
    end
    DistributedProgress(channel)
end

ProgressMeter.next!(pbar::DistributedProgress, args...; kwargs...) = put!(pbar.channel, (ProgressMeter.next!, args, kwargs))
ProgressMeter.update!(pbar::DistributedProgress, args...; kwargs...) = put!(pbar.channel, (ProgressMeter.update!, args, kwargs))
ProgressMeter.finish!(pbar::DistributedProgress, args...; kwargs...) = (put!(pbar.channel, nothing); close(pbar.channel))
