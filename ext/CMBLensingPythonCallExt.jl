
module CMBLensingPythonCallExt

using CMBLensing
using CMBLensing: extrapolate_Cℓs

if isdefined(Base, :get_extension)
    import PythonCall
else
    import ..PythonCall
end

CMBLensing.pyimport(x) = PythonCall.pyimport(x)
CMBLensing.PyArray(x) = PythonCall.PyArray(x)

end