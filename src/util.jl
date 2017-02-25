
export fieldvalues

import Base: ==
==(x,y,z,ws...) = x==y && ==(y,z,ws...)

@generated fieldvalues(x) = :((getfield(x,f) for f=fieldnames($x))...)
