using CMBLensing, JLD2, CodecZlib
Cℓ = camb(ℓmax=16000);
save(CMBLensing._default_Cℓs_path, "params", Cℓ.params, "Cℓ", Cℓ, compress=true)