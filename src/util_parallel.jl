
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
        transport = get(ENV,"JULIA_MPI_TRANSPORT","TCP")
    )
        
        if !MPI.Initialized()
            MPI.Init()
        end
        size = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)

        if size>1
            # workers don't return from this call:
            start_main_loop(
                Dict("TCP"=>TCP_TRANSPORT_ALL,"MPI"=>MPI_TRANSPORT_ALL)[transport],
                stdout_to_master=stdout_to_master,
                stderr_to_master=stderr_to_master
            )
            
            if @isdefined(CUDA) && CUDA.functional()
                assign_GPU_workers()
            else
                println(worker_info())
            end
        end

    end

    _mpi_rank() = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : nothing
    
end


"""
    assign_GPU_workers()

Assign each Julia worker process a unique GPU using `CUDA.device!`.
Workers may be distributed across different hosts, and each host can have
multiple GPUs.
"""
function assign_GPU_workers()
    @everywhere @eval Main using Distributed, CMBLensing
    master_uuid = @isdefined(CUDA) ? CUDA.uuid(device()) : nothing
    accessible_gpus = Dict(map(workers()) do id
        @eval Main @fetchfrom $id begin
            ds = CUDA.devices()
            # put master's GPU last so we don't double up on it unless we need to
            $id => sort((CUDA.deviceid.(ds) .=> CUDA.uuid.(ds)), by=(((k,v),)->v==$master_uuid ? Inf : k))
        end
    end)
    claimed = Set()
    assignments = Dict(map(workers()) do myid
        for (gpu_id, gpu_uuid) in accessible_gpus[myid]
            if !(gpu_uuid in claimed)
                push!(claimed, gpu_uuid)
                return myid => gpu_id
            end
        end
        error("Can't assign a unique GPU to every worker, process $myid has no free GPUs left.")
    end)
    @everywhere workers() device!($assignments[myid()])
    println(worker_info())
end


"""
    worker_info()

Returns string showing info about assigned workers.
"""
function worker_info()
    lines = @eval Main map(procs()) do id
        @fetchfrom id begin
            info = ["myid = $id"]
            !isnothing(CMBLensing._mpi_rank()) && push!(info, "mpi-rank = $(CMBLensing._mpi_rank())")
            push!(info, "host = $(gethostname())")
            @isdefined(CUDA) && push!(info, "device = $(sprint(io->show(io, MIME("text/plain"), CUDA.device()))) $(split(string(CUDA.uuid(CUDA.device())),'-')[1]))")
            " ("*join(info, ", ")*")"
        end
    end
    join(["Worker info:"; lines], "\n")
end
