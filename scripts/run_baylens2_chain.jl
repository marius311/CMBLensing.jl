using ArgParse, Match
s = ArgParseSettings()
@add_arg_table s begin
    "--configuration";
    "-T";                      default=Float32;          eval_arg=true
    "--Nside";                 default=128;              arg_type=Int
    "--θpix";                  default=3;                arg_type=Int
    "--pol";                   default=:P;               arg_type=Symbol
    "--beamFWHM";              default=1.;               arg_type=Float64
    "--μKarcminT";             default=1.;               arg_type=Float64
    "--bandpass_mask";         default="LowPass(3000)"
    "--pixel_mask_kwargs";     eval_arg=true
    "--storage";               default="Array";          range_tester=in(("Array","CuArray"))
    "--ϕstart";                default="quasi_sample";   range_tester=in(("quasi_sample","bestfit","0"))
    "--Nϕ_fac";                default=2f0;              arg_type=Float32
    "--filename"
    "--resume";                action=:store_true
    "--symp_kwargs";           default=[(N=25, ϵ=0.01)]; eval_arg=true
    "--nsamps_per_chain";      default=2000;             arg_type=Int
    "--nlenseflow_ode";        default=10;               arg_type=Int
    "--seed";                  arg_type=Int;
    "--nchunk";                default=30;               arg_type=Int
    "--nsavemaps";             default=5;                arg_type=Int
    "--nburnin_always_accept"; default=50;               arg_type=Int
    "--nburnin_fixθ";          default=10;               arg_type=Int
    "--sampled_params";        default=[:Aϕ,:r];         eval_arg=true
    "--rfid";                  default=0.05;             arg_type=Float64
end
args = parse_args(ARGS, s)

###

# configurations from the paper
merge!(args, @match args["configuration"] begin
    "2PARAM" => begin
        Dict(
            "pol"               => :P,
            "Nside"             => 256,
            "θpix"              => 2,
            "μKarcminT"         => 1/√2,
            "beamFWHM"          => 2,
            "bandpass_mask"     => "LowPass(5000)",
            "pixel_mask_kwargs" => (edge_padding_deg=0.4, apodization_deg=0.6, edge_rounding_deg=0.1, num_ptsrcs=0),
            "sampled_params"    => [:Aϕ,:r],
            "rfid"              => 0.04,
            "symp_kwargs"       => [(N=25, ϵ=0.02)]
        )
    end
    "MANY" => begin
        if !(args["rfid"] in [0,0.02,0.04])
            @warn "rfid=$(args["rfid"]) inconsistent with configuration $(args["configuration"])"
        end
        Dict(
            "pol"               => :P,
            "Nside"             => 256,
            "θpix"              => 3,
            "μKarcminT"         => 1/√2,
            "beamFWHM"          => 3,
            "bandpass_mask"     => "LowPass(3500)",
            "pixel_mask_kwargs" => (edge_padding_deg=0.6, apodization_deg=0.9, edge_rounding_deg=0.1, num_ptsrcs=0),
            "sampled_params"    => [:r],
            "symp_kwargs"       => [(N=25, ϵ=0.02)]
        )
    end
    "BIG" => begin
        if !(args["rfid"] in [0,0.01,0.02])
            @warn "rfid=$(args["rfid"]) inconsistent with configuration $(args["configuration"])"
        end
        Dict(
            "pol"               => :P,
            "Nside"             => 512,
            "θpix"              => 3,
            "μKarcminT"         => 1/√2,
            "beamFWHM"          => 3,
            "bandpass_mask"     => "LowPass(3500)",
            "pixel_mask_kwargs" => (edge_padding_deg=1.2, apodization_deg=1.8, edge_rounding_deg=0.2, num_ptsrcs=0),
            "sampled_params"    => [:r],
            "symp_kwargs"       => [(N=25, ϵ=0.02)]
        )
    end
    _::Nothing => Dict()
    c => error("Unrecognized configuration: $c")
end)

args["ϕstart"] = args["ϕstart"] == "0" ? 0 : Symbol(args["ϕstart"]);

### load dataset

@info "Loading code..."
@time begin
    if args["storage"]=="CuArray"
        using CuArrays
    end
    args["storage"] = eval(Symbol(args["storage"]))
    using CMBLensing
end


@info "Loading dataset..."
@unpack f, f̃, ϕ, ds = @time load_sim_dataset(
    Nside             = args["Nside"],
    T                 = args["T"],
    θpix              = args["θpix"], 
    pol               = args["pol"],
    L                 = LenseFlow{RK4Solver{args["nlenseflow_ode"]}},
    seed              = args["seed"],
    rfid              = args["rfid"],
    beamFWHM          = args["beamFWHM"],
    μKarcminT         = args["μKarcminT"],
    pixel_mask_kwargs = args["pixel_mask_kwargs"],
    bandpass_mask     = eval(Meta.parse(args["bandpass_mask"])),
    Nϕ_fac            = args["Nϕ_fac"],
);


### baylens2-specific Gibbs θ-pass 

using CMBLensing: grid_and_sample

function baylens2_gibbs_pass_θ(;kwargs...)
    
    @unpack f°,ϕ°,θ,ds,θrange,progress = kwargs
    
    lnP_Aϕ, θ_Aϕ, lnP_r, θ_r = (), (), (), ()
    
    # sample Aϕ
    if :Aϕ in keys(θrange)
        lnP_Aϕ, θ_Aϕ = grid_and_sample(
            θ′->lnP(:mix,f°,ϕ°,merge(θ,θ′),ds), 
            (Aϕ=θrange.Aϕ,), 
            progress=(progress==:verbose)
        )
        lnP_Aϕ = (Aϕ=lnP_Aϕ,)
        θ = merge(θ,θ_Aϕ)
    end
    
    # sample r
    if :r in keys(θrange)
        lnP_r, θ_r = grid_and_sample(
            θ′->lnP(:mix,f°,ϕ°,merge(θ,θ′),ds), 
            (r=θrange.r,), 
            progress=(progress==:verbose)
        )
        lnP_r = (r=lnP_r,)
        θ = merge(θ,θ_r)
    end
        
    (;lnP_Aϕ..., lnP_r...), θ
    
end



## run sampler

θrange₀ = (
    Aϕ = range(0.75, 1.25, length=50), 
    r  = range(sqrt(1e-6), sqrt(1e-1), length=50).^2
)

if !isinteractive()
    
    @info "Starting chain..."
    @unpack (rundat, chains) = sample_joint(
        
        ds,
        
        nchains = 1,
        
        nsamps_per_chain      = args["nsamps_per_chain"],
        storage               = args["storage"],
        chains                = args["resume"] ? :resume : nothing,
        nchunk                = args["nchunk"],
        nsavemaps             = args["nsavemaps"],
        nburnin_always_accept = args["nburnin_always_accept"],
        nburnin_fixθ          = args["nburnin_fixθ"],
        
        θrange = (;(k=>v for (k,v) in pairs(θrange₀) if k in args["sampled_params"])...),
        gibbs_pass_θ = baylens2_gibbs_pass_θ,

        ϕstart = args["ϕstart"],
        Nϕ_fac = args["Nϕ_fac"],

        wf_kwargs   = (tol=1e-1, nsteps=200),
        symp_kwargs = args["symp_kwargs"],
        MAP_kwargs  = (αmax=0.3, nsteps=5),
        
        progress = :summary,
        interruptable = true,
        
        filename = args["filename"]
        
    );
    
end
