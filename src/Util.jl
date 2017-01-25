module Util

export fieldvalues

@generated fieldvalues(x) = :((getfield(x,f) for f=fieldnames($x))...)

end
